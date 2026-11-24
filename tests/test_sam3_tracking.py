"""Regressions for persistent mask IDs, isolated poses and safe tracking replacement."""
import copy
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT,os.path.join(ROOT,'pipeline')]
import automark
import sam3_tracking as SAM
import run_pipeline as RP
import quality
from common import jsave,jload


def bundle(records, shape=(20,40)):
    masks = []
    for _,pid in records:
        mask = np.zeros(shape,np.uint8)
        mask[4:16,4:16] = 1 if pid==1 else 0
        if pid!=1: mask[4:16,24:36] = 1
        masks.append(np.packbits(mask.ravel()))
    return {'frames':np.array([f for f,_ in records],np.int32),'ids':np.array([p for _,p in records],np.int64),
            'packed':np.asarray(masks,np.uint8).reshape(len(records),-1),'shape':np.array(shape),'version':1}


class FakePose:
    engine='cuda'
    def __init__(self): self.images=[]
    def poses(self,items):
        result=[]
        for image,boxes in items:
            self.images.append(image.copy())
            box=boxes[0]; center=(np.array(box[:2])+box[2:])/2
            keys=np.tile(center,(1,17,1)); scores=np.ones((1,17))*.9
            keys[0,0]=[39,19]  # Deliberately place one joint on a neighbour/background.
            result.append((keys,scores))
        return result


class MaskTests(unittest.TestCase):
    def test_mask_bitpacking_and_resizing(self):
        mask=np.zeros((5,7),np.uint8); mask[1:4,2:5]=1
        b={'frames':np.array([0]),'ids':np.array([8]),'packed':np.packbits(mask.ravel())[None],
           'shape':np.array(mask.shape),'version':1}
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'m.npz'); np.savez_compressed(path,**b)
            restored=SAM.read_masks(path)
            np.testing.assert_array_equal(SAM.mask_at(restored,0,7,5),mask)
            self.assertEqual(SAM.mask_at(restored,0,14,10).sum(),36)

    def test_duplicate_identity_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'m.npz'); np.savez_compressed(path,**bundle([(0,1),(0,1)]))
            with self.assertRaises(ValueError): SAM.read_masks(path)

    def test_union_uses_frame_and_persistent_id(self):
        b=bundle([(0,1),(0,2),(2,1)])
        self.assertEqual(SAM.union_at(b,0,40,20).sum(),288)
        self.assertEqual(SAM.union_at(b,0,40,20,{1}).sum(),144)
        self.assertFalse(SAM.union_at(b,1,40,20).any())

    def test_pose_order_never_changes_mask_identity(self):
        b=bundle([(0,1),(0,2),(1,2),(1,1)])
        frames=[np.full((30,40,3),200,np.uint8) for _ in range(2)]
        marker=FakePose(); tracks=SAM.measure_poses(b,frames,5,25,marker)
        self.assertEqual(set(tracks),{1,2})
        self.assertEqual(set(tracks[1]),{0,1})
        np.testing.assert_allclose(tracks[1][0][0][1],[10,15])
        np.testing.assert_allclose(tracks[1][1][0][1],[10,15])
        np.testing.assert_allclose(tracks[2][1][0][1],[30,15])

    def test_pose_crop_cannot_see_neighbour_and_outside_joints_are_downweighted(self):
        marker=FakePose(); b=bundle([(0,1)])
        frames=[np.full((30,40,3),230,np.uint8)]
        tracks=SAM.measure_poses(b,frames,5,25,marker)
        self.assertTrue((marker.images[0][5,25]==128).all())
        self.assertTrue((marker.images[0][5,5]==230).all())
        self.assertLess(tracks[1][0][1][0],.3)
        self.assertGreater(tracks[1][0][1][1],.8)
        self.assertEqual(tracks[1][0][2],[4.,9.,16.,21.])

    def test_empty_mask_frames_are_not_fabricated(self):
        tracks=SAM.measure_poses(bundle([(0,1),(2,1)]),[np.zeros((20,40,3),np.uint8)]*3,0,20,FakePose())
        self.assertEqual(set(tracks[1]),{0,2})


class PersistentIdentityTests(unittest.TestCase):
    def test_every_configured_person_colour_is_supported_by_preview(self):
        from common import DEFAULT_CFG
        self.assertFalse(set(DEFAULT_CFG['colours'])-set(RP.PALETTE))

    def test_lazy_frame_folder_accepts_unicode_paths(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder=pathlib.Path(tmp)/'人物'; folder.mkdir()
            Image.new('RGB',(40,20)).save(folder/'000000.jpg')
            source=SAM.FrameFolder(str(folder),1008,(.5,)*3,(.5,)*3)
            self.assertEqual((source.width,source.height,len(source)),(40,20,1))

    def test_geometry_cleanup_cannot_merge_or_fill_sam3_tracks(self):
        from dedupe_roles import duplicates
        from test_agent import TestStitch
        pose={'tracker_backend':'sam3','people':[TestStitch.track(1,0,40,500),TestStitch.track(2,20,60,500)]}
        before=copy.deepcopy(pose)
        self.assertEqual(RP.stitch_tracks(pose,[(0,60)]),{})
        self.assertEqual(RP.fill_track_gaps(pose,[(0,60)]),{})
        self.assertEqual(duplicates(pose,{'1','2'}),[])
        self.assertEqual(pose,before)

    def test_mask_confirmed_partial_person_survives_weak_pose_filter(self):
        people={1:([0,1,2,3],np.zeros((4,17,3)))}
        self.assertEqual(quality.drop_junk(people,{'_persistent_tracks':True},30),people)
        self.assertEqual(quality.drop_junk(people,{},30),{})

    def test_pose_slice_preserves_mask_id_and_metadata(self):
        pose={'tracker_backend':'sam3','masks_file':'sam3_masks.npz','people':[
            {'id':1,'sam3_id':8,'frames':{'4':[[1,2,.8]]*17},'boxes':{'4':[0,0,10,20]}}]}
        sliced=RP.slice_pose(pose,4,8)
        self.assertEqual(sliced['people'][0]['sam3_id'],8)
        self.assertIn('0',sliced['people'][0]['frames'])
        self.assertEqual(sliced['masks_file'],'sam3_masks.npz')

    def test_cache_invalidates_backend_checkpoint_and_tracking_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt=pathlib.Path(tmp)/'sam3.pt'; ckpt.write_bytes(b'checkpoint')
            c={'tracker_backend':'sam3','sam3_checkpoint':str(ckpt)}
            pose={'tracker_backend':'sam3','tracking_signature':SAM.signature(c)}
            self.assertFalse(SAM.needs_tracking(pose,c))
            self.assertTrue(SAM.needs_tracking({},c))
            self.assertTrue(SAM.needs_tracking(pose,dict(c,sam3_probability=.2)))
            ckpt.write_bytes(b'new checkpoint')
            self.assertTrue(SAM.needs_tracking(pose,c))
            self.assertTrue(SAM.needs_tracking(pose,{'tracker_backend':'geometry'}))


class ReplacementTests(unittest.TestCase):
    def prepare(self,tmp):
        old=os.path.join(tmp,'old'); staged=os.path.join(tmp,'new')
        for folder in (old,staged): os.makedirs(os.path.join(folder,'review'))
        jsave(os.path.join(old,'pose.json'),{'tracker_backend':'geometry','people':[{'id':1}]})
        jsave(os.path.join(old,'checked.json'),{'people':{'1':{'0':'manual edit'}}})
        jsave(os.path.join(old,'review.json'),{'confirmed':True,'people':{'1':{'role':'main'}},'camera':{'type':'static'}})
        jsave(os.path.join(old,'hints.json'),{'people':{'1':{'metric_scale':1.2}},'camera':{'height':2.}})
        jsave(os.path.join(old,'ai.json'),{'people':{'1':{}}})
        jsave(os.path.join(staged,'pose.json'),{'tracker_backend':'sam3','people':[{'id':2,'sam3_id':7}]})
        jsave(os.path.join(staged,'tracking_report.json'),{'backend':'sam3'})
        pathlib.Path(staged,'check.html').write_text('new editor',encoding='utf-8')
        pathlib.Path(staged,'sam3_masks.npz').write_bytes(b'new masks')
        return old,staged

    def test_publish_preserves_edits_in_backup_and_resets_old_id_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,staged=self.prepare(tmp); backup=SAM.publish(staged,old)
            self.assertTrue(os.path.isfile(os.path.join(backup,'checked.json')))
            self.assertEqual(jload(os.path.join(backup,'review.json'))['people']['1']['role'],'main')
            self.assertFalse(os.path.exists(os.path.join(old,'checked.json')))
            review=jload(os.path.join(old,'review.json'))
            self.assertFalse(review['confirmed']); self.assertEqual(review['people'],{})
            self.assertEqual(review['camera'],{'type':'static'})
            self.assertEqual(jload(os.path.join(old,'hints.json')),{'camera':{'height':2.}})
            self.assertEqual(jload(os.path.join(old,'pose.json'))['tracker_backend'],'sam3')

    def test_failed_sam_inference_preserves_previous_tracking_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,staged=self.prepare(tmp)
            path=os.path.join(tmp,'old.mp4')
            args=types.SimpleNamespace(base=None,force=True)
            old_pose=pathlib.Path(old,'pose.json').read_bytes()
            frames=[np.full((20,40,3),200,np.uint8)]*4
            with patch.object(automark,'cfg',return_value={'tracker_backend':'sam3'}), \
                 patch.object(automark,'read_frames',return_value=(frames,30.)), \
                 patch.object(SAM,'run_masks',side_effect=RuntimeError('GPU failure')):
                with self.assertRaisesRegex(RuntimeError,'GPU failure'):
                    automark.process(path,tmp,args,None)
            self.assertEqual(pathlib.Path(old,'pose.json').read_bytes(),old_pose)
            self.assertTrue(jload(os.path.join(old,'review.json'))['confirmed'])
            self.assertTrue(os.path.exists(os.path.join(old,'checked.json')))
            self.assertFalse(os.path.exists(os.path.join(old,'tracking_backups')))

    def test_missing_checkpoint_fails_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(SAM.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'checkpoint missing'):
                SAM.run_masks([np.zeros((20,40,3),np.uint8)],0,20,[(0,1)],tmp,{'sam3_checkpoint':os.path.join(tmp,'missing.pt')})
            run.assert_not_called()

    def test_incomplete_staged_tracking_cannot_reset_previous_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,staged=self.prepare(tmp)
            os.remove(os.path.join(staged,'pose.json'))
            with self.assertRaisesRegex(RuntimeError,'Incomplete tracking output'): SAM.publish(staged,old)
            self.assertTrue(jload(os.path.join(old,'review.json'))['confirmed'])
            self.assertTrue(os.path.isfile(os.path.join(old,'checked.json')))
            self.assertFalse(os.path.isdir(os.path.join(old,'tracking_backups')))


if __name__=='__main__': unittest.main()
