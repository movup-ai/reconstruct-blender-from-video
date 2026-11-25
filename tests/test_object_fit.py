"""Image/depth fitting, partial furniture, semantic rooms and final geometry validation."""
import copy
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import cv2
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
import object_fit as O
import env_auto as E
import props
import profiling
from common import jload


def observation(obj,frame=0,center=(0,0,1.4),parts=None):
    R=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    view={'frame':frame,'shape':[144,192],'K':[[180.,0,96],[0,180.,72],[0,0,1]],'R':R.tolist(),
          'C':list(center),'depth':obj['pos'][1]-.2,'depth_sigma':.4,'partial':bool(parts),'parts':parts,'source_scale':4.}
    view['occluder']=O.bits(np.zeros((144,192),bool))
    mask=O.raster(obj,view);view['mask']=O.bits(mask)
    return view


class ObjectFitTests(unittest.TestCase):
    def test_drawn_tree_keeps_its_extent_despite_bad_later_masks(self):
        actual={'type':'pine','pos':[-1.8,8.,0.],'size':[.9,3.5,.9],'yaw':0}
        view=observation(actual)
        points=np.concatenate([v for _,v,_ in O._parts('pine',actual['size'])])+actual['pos']
        view.update(primary=True,annotation_box=O.bounds(points,view).tolist())
        candidate=dict(actual,pos=[-1.,8.,1.],size=[.4,1.2,.4],_observations=[view])
        O.seed_annotation(candidate,view)
        bad=dict(view,frame=2,primary=False);bad.pop('annotation_box')
        mask=O.unbits(view['mask'],view['shape']);bad['mask']=O.bits(np.roll(mask,55,axis=1))
        candidate['_observations'].append(bad)
        O.fit(candidate,max_evaluations=30)
        points=np.concatenate([v for _,v,_ in O._parts('pine',candidate['size'])])
        q=np.r_[candidate['pos'],np.log(candidate['size']),np.radians(candidate['yaw'])]
        world=O._transform(points/np.asarray(candidate['size'])[[0,2,1]],q)
        self.assertLess(np.abs(O.box_error(world,view)).max(),.12)
        self.assertGreater(candidate['size'][1],2.8)
        self.assertLessEqual(candidate['_fit']['silhouette_after'],candidate['_fit']['silhouette_before']+1e-8)

    def test_invalid_depth_does_not_crop_the_object_silhouette(self):
        shape=(60,80);mask=np.zeros(shape,bool);mask[5:55,10:30]=True
        valid=np.zeros(shape,bool);valid[30:]=True
        fr=types.SimpleNamespace(H=60,W=80,P=np.dstack([np.zeros(shape),np.zeros(shape),np.full(shape,7.)]),
            person=np.zeros(shape,bool),bars=np.zeros(shape,bool),valid=valid,s=1.,fpix=80.,Rw=np.eye(3),Cl=np.zeros(3),k=1.,f=0)
        view=O.observation(fr,mask)
        np.testing.assert_array_equal(O.unbits(view['mask'],shape),mask)
        self.assertEqual(view['depth'],7.)

    def test_visible_tree_base_reconciles_wrong_foliage_depth_with_the_ground(self):
        actual={'type':'palm','pos':[0,8.,0.],'size':[.8,3.5,.8],'yaw':0}
        view=observation(actual);points=np.concatenate([v for _,v,_ in O._parts('palm',actual['size'])])+actual['pos']
        view.update(primary=True,annotation_box=O.bounds(points,view).tolist(),depth=18.,ground_plane=[0.,0.,0.])
        candidate=dict(actual,size=[.8,1.,.8],_observations=[view])
        O.seed_annotation(candidate,view);O.fit(candidate,max_evaluations=25)
        self.assertTrue(view['ground_contact']);self.assertEqual(view['measured_depth'],18.)
        self.assertLess(abs(candidate['pos'][2]),.05)
        self.assertLess(abs(candidate['pos'][1]-8.),1.)
        self.assertGreater(candidate['size'][1],3.)

    def test_tree_contact_uses_the_actual_terrain_instead_of_flat_zero(self):
        actual={'type':'pine','pos':[0,8.,.6],'size':[.8,3.5,.8],'yaw':0}
        view=observation(actual);points=np.concatenate([v for _,v,_ in O._parts('pine',actual['size'])])+actual['pos']
        terrain={'z':.5,'heights':[[0.,8.,.1]],'falloff':3.}
        view.update(primary=True,annotation_box=O.bounds(points,view).tolist(),ground_plane=[0.,0.,0.],ground_definition=terrain)
        candidate=dict(actual,pos=[0,8,0],_observations=[view]);O.seed_annotation(candidate,view);O.fit(candidate,max_evaluations=25)
        from build_env import Ground
        self.assertLess(abs(candidate['pos'][2]-Ground(terrain).at(*candidate['pos'][:2])),.03)
        self.assertGreater(candidate['pos'][2],.5)

    def test_refinement_never_replaces_a_better_starting_fit(self):
        actual={'type':'palm','pos':[0,7.,.1],'size':[.8,3.,.8],'yaw':0}
        view=observation(actual);candidate=dict(actual,_observations=[view])
        O.fit(candidate,max_evaluations=15)
        self.assertLessEqual(candidate['_fit']['silhouette_after'],candidate['_fit']['silhouette_before']+1e-8)

    def test_edit_replaces_an_automatic_tree_from_another_sampled_frame(self):
        tree={'type':'pine','pos':[-.5,7.,0.],'size':[.8,3.,.8],'yaw':0}
        view=observation(tree);points=np.concatenate([v for _,v,_ in O._parts('pine',tree['size'])])+tree['pos']
        view['annotation_box']=O.bounds(points,view).tolist()
        edited=dict(tree,_observations=[view]);automatic=dict(tree,_src={'frame':29})
        self.assertTrue(O.annotation_replaces(automatic,edited))
        self.assertFalse(O.annotation_replaces(dict(automatic,pos=[3.,7.,0.]),edited))
        self.assertFalse(O.annotation_replaces(dict(automatic,type='chair'),edited))

    def test_drawn_fallback_is_not_silently_uniformly_shrunk_by_the_builder(self):
        from build_env import build,Cam
        data={'frames':1,'res':[192,144],'lens':34.,'sensor':36.,'cam':[[0,0,1.4,90,0,0]],'zoom':[1.]}
        obj={'name':'drawn','type':'pine','bbox':[10,5,30,120],'dist':4.,'frame':0,
             '_src':{'cls':'drawn','frame':0,'px':[10,5,30,120]}}
        with patch('geometry_protection.match_source_box',side_effect=AssertionError('legacy uniform shrink')):
            self.assertTrue(build({'ground':{'size':0},'objects':[obj]},Cam(data)))

    def test_explicit_distance_scales_fitted_placement_and_dimensions_together(self):
        tree={'type':'pine','pos':[1.,6.,0.],'size':[.8,3.,.8],'dist':6.,'_C':[0.,0.,1.5],
              '_reach':[6.,7.], '_observations':[{'depth':6.,'depth_sigma':.5}]}
        E.push_back(tree,2.)
        np.testing.assert_allclose(tree['pos'],[2.,12.,-1.5])
        np.testing.assert_allclose(tree['size'],[1.6,6.,1.6])
        self.assertEqual(tree['_observations'][0]['depth'],12.)

    def test_image_flow_matches_the_same_chair_with_uncertain_camera(self):
        rng=np.random.default_rng(4);first=rng.integers(0,256,(144,192,3),dtype=np.uint8)
        second=cv2.warpAffine(first,np.float32([[1,0,9],[0,1,2]]),(192,144))
        actual={'type':'chair','pos':[0,4,.1],'size':[.45,.9,.5],'yaw':0}
        view=observation(actual);old=np.zeros((144,192),bool);old[50:85,65:83]=True;view['mask']=O.bits(old)
        moved=cv2.warpAffine(old.astype(np.uint8),np.float32([[1,0,9],[0,1,2]]),(192,144))>0
        previous=dict(actual,_observations=[view]);new=dict(view,frame=1,mask=O.bits(moved))
        candidate=dict(actual,_observations=[new])
        self.assertIs(O.match([previous],candidate,{0:first,1:second}),previous)
        wrong=dict(new,mask=O.bits(np.roll(moved,35,axis=1)))
        self.assertIsNone(O.match([previous],dict(actual,_observations=[wrong]),{0:first,1:second}))

    def test_width_height_depth_axes_match_exported_mesh(self):
        dims=np.array([.8,1.6,.3]);q=np.r_[[0,4,.2],np.log(dims),.4]
        v=np.array([[-.4,-.15,0],[.4,.15,1.6]])
        world=O._transform(v/dims[[0,2,1]],q)
        self.assertAlmostEqual(np.ptp(world[:,2]),1.6)
        np.testing.assert_allclose(O._transform(v/dims[[0,2,1]],np.r_[[0,0,0],np.log(dims),0]),v)

    def test_partial_chair_joint_fit_preserves_complete_height_and_aligns_back(self):
        actual={'type':'chair','pos':[-.3,4.5,.1],'size':[.45,.9,.5],'yaw':20}
        view=observation(actual,parts=['back'])
        candidate=dict(actual,pos=[-.05,4.7,.22],size=[.6,.85,.6],yaw=-5,_observations=[view])
        initial=O.metrics(O.raster(candidate,view),view)
        O.fit(candidate,max_evaluations=35)
        final=O.metrics(O.raster(candidate,view),view)
        self.assertGreater(final['iou'],initial['iou']+.15)
        self.assertLess(final['center_error_px'],8)
        self.assertGreaterEqual(candidate['size'][1],.6)
        self.assertTrue(candidate['_fit']['partial'])

    def test_joint_fit_uses_three_views_and_improves_visible_shape(self):
        actual={'type':'box','pos':[.2,5.,.5],'size':[1.,1.3,.7],'yaw':25}
        views=[observation(actual,f,(x,0,1.4)) for f,x in enumerate((-.4,0,.4))]
        candidate=dict(actual,pos=[.4,5.3,.35],size=[1.2,1.,.9],yaw=0,_observations=views)
        before=np.mean([O.metrics(O.raster(candidate,v),v)['iou'] for v in views])
        O.fit(candidate,max_evaluations=40)
        after=[O.metrics(O.raster(candidate,v),v) for v in views]
        self.assertGreater(np.mean([v['iou'] for v in after]),before+.2)
        self.assertTrue(all(v['center_error_px']<10 for v in after))
        self.assertEqual(candidate['_fit']['observations'],3)

    def test_locked_dimensions_are_preserved_during_joint_placement(self):
        actual={'type':'plant_pot','pos':[0,4.,.1],'size':[.5,1.,.5],'yaw':0}
        view=observation(actual)
        candidate=dict(actual,pos=[.1,4.1,.15],_observations=[view],size_locked=True)
        O.fit(candidate,max_evaluations=20)
        np.testing.assert_array_equal(candidate['size'],actual['size'])

    def test_final_exported_geometry_is_audited_not_only_fitter_parameters(self):
        actual={'type':'box','pos':[0,5.,.2],'size':[1.,1.,.5],'yaw':0}
        view=observation(actual);parts=props.place(props.build('box',1.,1.,.5),(0,5),0,.2)
        meshes=[{'name':'test_'+n,'v':v,'f':f,'_observations':[view],'_primitive':'test','kind':'box'} for n,v,f,_ in parts]
        data={'meshes':meshes}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(O.audit_scene(data,tmp)['passed'])
            for m in meshes:m['v']=(np.asarray(m['v'])+[1.,0,0]).tolist()
            self.assertFalse(O.audit_scene(data,tmp)['passed'])

    def test_concave_wall_triangulation_does_not_fill_missing_corner(self):
        p=np.array([[0,0],[4,0],[4,1],[1,1],[1,4],[0,4]])
        faces=O.triangulate(p)
        area=sum(abs((p[b,0]-p[a,0])*(p[c,1]-p[a,1])-(p[b,1]-p[a,1])*(p[c,0]-p[a,0]))/2 for a,b,c in faces)
        self.assertAlmostEqual(area,7.)


class FurnitureAndRoomTests(unittest.TestCase):
    def test_plant_and_pot_share_one_semantic_piece_without_filling_other_objects(self):
        labels=np.zeros((100,100),np.int16);labels[20:50,30:60]=1;labels[50:65,38:52]=2
        fr=types.SimpleNamespace(sem=labels,segm=types.SimpleNamespace(names=['wall','plant','pot']),
                                valid=np.ones(labels.shape,bool),person=np.zeros(labels.shape,bool),bars=np.zeros(labels.shape,bool))
        def label(seg,mask):return ('plant',float((seg[mask]==1).mean()))
        fr.segm.label=label
        pieces=E.soft_parts(fr)
        self.assertEqual(len(pieces),1);self.assertEqual(pieces[0][1],'plant')
        np.testing.assert_array_equal(pieces[0][0],labels>0)
        self.assertEqual(__import__('prototypes').vegetation_kind('plant',True),'plant_pot')

    def test_small_elevated_confident_chair_is_retained(self):
        rng=np.random.default_rng(3);P=rng.random((100,3))*[.3,.1,.3]+[0,4.,1.2]
        region={'cls':'object','P':P,'Pabs':P,'z':(1.2,1.35,1.5),'rng':4.,'area':.008,
                'ptype':'chair','label':('chair',.96),'bbox_px':[10,10,30,40]}
        result=E.make_object(region,np.array([0,0,1.5]),True,types.SimpleNamespace(f=0,k=1),100,'chair')
        self.assertIsNotNone(result);self.assertEqual(result['type'],'chair')

    def test_invalid_depth_does_not_turn_office_into_outdoors(self):
        names=['ceiling','windowpane','table','chair','building'];labels=np.tile(np.arange(5),(20,4))
        fr=types.SimpleNamespace(sem=labels,segm=types.SimpleNamespace(names=names),
                                bars=np.zeros(labels.shape,bool),person=np.zeros(labels.shape,bool),valid=np.zeros(labels.shape,bool))
        indoor,evidence=O.room(fr)
        self.assertTrue(indoor);self.assertEqual(evidence['source'],'semantic room surfaces')
        fr.segm.names=['sky','tree','earth','building','chair'];self.assertFalse(O.room(fr)[0])

    def test_potted_prototype_has_connected_pot_stem_and_round_canopy(self):
        parts=props.build('plant_pot',.8,1.3,.8)
        self.assertEqual({p[0] for p in parts},{'pot','stem','leaves'})
        by={name:np.asarray(v) for name,v,*_ in parts}
        self.assertGreaterEqual(by['stem'][:,2].max(),by['leaves'][:,2].min())
        self.assertLessEqual(by['stem'][:,2].min(),by['pot'][:,2].max()+1e-7)


class DetailedTimingTests(unittest.TestCase):
    def test_cached_worker_profile_is_not_reported_as_a_new_run(self):
        @profiling.profile_clip
        def clip(video,config,args):return {'status':'ok'}
        with tempfile.TemporaryDirectory() as tmp:
            work=pathlib.Path(tmp)/'clip';work.mkdir()
            from common import jsave
            jsave(str(work/'environment_profile.json'),{'stages':{'sam_inference':{'seconds':99}}})
            clip('clip.mp4',{'work_root':tmp},types.SimpleNamespace(name='clip',stage='all'))
            self.assertNotIn('details',jload(str(work/'profile_all.json')))

    def test_worker_profile_records_loading_inference_fitting_and_cache_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'environment_profile.json')
            with profiling.recording(path):
                profiling.add_seconds('model_loading',1.2)
                profiling.add_seconds('sam_inference',.4,2)
                profiling.add_seconds('prototype_fitting',.3)
                profiling.counter('sam_cache_hits',3)
            result=jload(path)
            self.assertEqual(result['stages']['sam_inference']['calls'],2)
            self.assertEqual(result['counters']['sam_cache_hits'],3)
            self.assertIn('total_seconds',result)


if __name__=='__main__':unittest.main()
