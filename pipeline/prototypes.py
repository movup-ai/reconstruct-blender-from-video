"""Recognizable vegetation prototypes sized from measured regions, not noisy relief meshes."""
import re
import numpy as np


def vegetation_kind(label, indoor=False):
    text = str(label or '').lower()
    if re.search(r'\b(potted|houseplant|house plant|plant pot|flowerpot)\b', text): return 'plant_pot'
    if text.strip()=='pot':return 'pot'
    words = set(re.findall(r'[a-z]+', text))
    def has(*terms): return any(t in words or t+'s' in words for t in terms)
    if has('forest', 'woodland', 'grove', 'woods'): return 'forest'
    if has('palm'): return 'palm'
    if has('bush', 'shrub', 'bushes'): return 'bush'
    if has('hedge', 'hedgerow'): return 'hedge'
    if has('tree', 'pine', 'fir', 'spruce', 'conifer'): return 'pine'
    if has('plant', 'flower', 'foliage', 'leaves'):
        return 'plant_pot' if indoor else 'bush'
    return None


def measured_object(kind, points, name, source):
    p = np.asarray(points, float)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 8: return None
    lo, hi = np.percentile(p, [5, 95], axis=0)
    width, depth = np.maximum(hi[:2]-lo[:2], .08)
    height = max(.12, float(hi[2]-lo[2]))
    if kind in ('pine', 'palm'):
        depth = max(depth, .35 * width)
    elif kind in ('bush', 'hedge'):
        height = min(height, max(.25, width*1.2)); depth = max(depth, width*.5)
    center = (lo+hi)/2
    return {'name': name, 'type': kind, 'prototype': kind,
            'pos': [float(center[0]), float(center[1]), float(lo[2])],
            'size': [float(width), height, float(depth)], 'yaw': 0., 'colour': 'tree', '_src': source}
