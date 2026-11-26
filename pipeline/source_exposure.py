"""Conservative, shot-local exposure changes measured outside tracked people.

This measures coherent background dimming, not absolute light direction or HDR
calibration. Ambiguous/insufficient background evidence leaves exposure unchanged.
"""
import cv2
import numpy as np


def measure(frames, shots, pose, y0, height, fps=30.):
    if not frames:
        return {'stops': [], 'shots': [], 'source': 'no frames'}
    width = frames[0].shape[1]
    sw, sh = 160, max(48, int(height * 160 / width))
    samples = []
    tracks = pose.get('people', [])
    for f, frame in enumerate(frames):
        image = cv2.resize(frame[y0:y0+height], (sw, sh)).astype(float) / 255.
        # Linearize sRGB before taking luminance ratios (input is BGR).
        linear = np.where(image <= .04045, image / 12.92, ((image + .055) / 1.055) ** 2.4)
        luminance = linear @ np.array([.0722, .7152, .2126])
        valid = np.ones((sh, sw), bool)
        for track in tracks:
            k = np.asarray(track.get('frames', {}).get(str(f), []), float)
            if k.shape != (17, 3):
                continue
            k = k[k[:, 2] > .15]
            if len(k) < 2:
                continue
            xy = (k[:, :2] - [0, y0]) * [sw / width, sh / height]
            lo, hi = xy.min(0), xy.max(0)
            margin = np.maximum([6., 8.], (hi-lo) * [.4, .25])
            x0, y1 = np.maximum(0, np.floor(lo-margin)).astype(int)
            x1, y2 = np.minimum([sw, sh], np.ceil(hi+margin)).astype(int)
            valid[y1:y2, x0:x1] = False
        row = []
        for yy in np.array_split(np.arange(sh), 6):
            for xx in np.array_split(np.arange(sw), 8):
                mask = valid[np.ix_(yy, xx)]
                tile = luminance[np.ix_(yy, xx)]
                row.append(float(np.median(tile[mask])) if mask.mean() >= .8 else np.nan)
        samples.append(row)
    return from_tiles(np.asarray(samples), shots, fps)


def from_tiles(samples, shots, fps=30.):
    samples = np.asarray(samples, float)
    stops = np.zeros(len(samples)); reports = []
    for first, end in shots:
        values = samples[first:end]
        usable = np.isfinite(values).mean(0) >= .7
        if usable.sum() < 6:
            reports.append({'frames': [first, end-1], 'applied': False, 'reason': 'insufficient background tiles'})
            continue
        values = values[:, usable]
        baseline = np.nanpercentile(values, 90, axis=0)
        keep = baseline > .01
        if keep.sum() < 6:
            reports.append({'frames': [first, end-1], 'applied': False, 'reason': 'background too dark to calibrate'})
            continue
        ratios = np.log2(np.maximum(values[:, keep], .0001) / baseline[keep])
        curve = np.zeros(len(values)); reliable = np.zeros(len(values), bool)
        for i, row in enumerate(ratios):
            finite = row[np.isfinite(row)]
            if len(finite) < 6:
                continue
            median = np.median(finite)
            # Real exposure changes agree over multiple static-background regions.
            reliable[i] = np.median(np.abs(finite-median)) <= .3
            if reliable[i]:
                curve[i] = np.clip(median, -8., 0.)
        applied = bool(reliable.mean() >= .7 and np.min(curve) < -.6)
        edges=np.flatnonzero(np.diff(np.r_[False,~reliable,False]))
        longest_gap=int(np.max(edges[1::2]-edges[::2])) if len(edges) else 0
        stable=longest_gap<=max(2,round(.15*fps))
        applied=bool(applied and stable)
        if applied:
            # A brief uncertain estimate must not flash the entire scene back to
            # neutral. Interpolate short gaps; ambiguous long gaps reject the shot.
            stops[first:end] = np.interp(np.arange(len(curve)),np.flatnonzero(reliable),curve[reliable])
        reports.append({'frames': [first, end-1], 'applied': applied,
                        'reliable_fraction': round(float(reliable.mean()), 3),
                        'min_stops': round(float(curve.min()), 3),
                        'max_unreliable_gap':longest_gap,
                        'reason': 'coherent background dimming' if applied else
                            ('ambiguous brightness interval' if not stable else 'no reliable substantial exposure change')})
    return {'stops': stops.round(4).tolist(), 'shots': reports,
            'source': 'linearized source background tiles excluding tracked people; relative stops per shot'}


def attach(data, video, work, config, pose=None, info=None, frames=None):
    """Render-only, opt-in analysis; switching it never requires a new human fit."""
    if not config.get('source_exposure',False):
        data.pop('source_exposure',None)
        return
    from pathlib import Path
    from common import jload,read_frames
    import validation as V
    if info is None:info=jload(str(Path(work)/'clip_info.json'))
    if pose is None:
        from run_pipeline import marks_dir
        pose=jload(str(Path(marks_dir(video,config))/'pose.json'))
    shots=[tuple(s['frames']) for s in data.get('shots',[])] or [(0,data['frames']-1)]
    key=V.digest({'video':V.file_digest(video),'tracks':[p.get('frames',{}) for p in pose.get('people',[])],
        'crop':[info['y0'],info['Hpic']],'shots':shots,'fps':info['fps'],
        'implementation':V.implementation(('source_exposure.py',))})
    def compute():
        pictures=frames if frames is not None else read_frames(video)[0]
        return measure(pictures,[(a,b+1) for a,b in shots],pose,info['y0'],info['Hpic'],info['fps'])
    curve=V.cached_report(Path(work)/'source_exposure.json',key,compute)
    if len(curve['stops'])!=data['frames']:raise ValueError('Exposure curve frame count does not match the scene')
    data['source_exposure']=curve
