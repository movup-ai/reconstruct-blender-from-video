"""Automatic shrub selection, stable bounded prototypes and local fitting recovery."""
import pathlib,sys,types,unittest,copy,tempfile
import cv2
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT),str(ROOT/'tests')]
import prototypes as P
import props
import object_fit as O
import env_auto as E
import sam3_scene as SAM
from test_object_fit import observation


class AutomaticShrubTests(unittest.TestCase):
    def points(self,height=1.,width=2.,base=.05):
        return np.random.default_rng(3).random((500,3))*[width,.5,height]+[-width/2,6.,base]

    def mask(self,wide=False):
        mask=np.zeros((200,300),np.uint8)
        cv2.ellipse(mask,(150,150),(110 if wide else 45,25 if wide else 35),0,0,360,1,-1)
        return mask>0

    def test_generic_tree_label_becomes_bush_for_low_compact_foliage(self):
        kind,evidence=P.automatic_kind('tree',self.points(),self.mask())
        self.assertEqual(kind,'bush');self.assertIn('low compact',evidence['rule'])

    def test_low_wide_row_becomes_hedge_but_tall_canopy_stays_tree(self):
        self.assertEqual(P.automatic_kind('tree',self.points(width=5),self.mask(True))[0],'hedge')
        self.assertEqual(P.automatic_kind('tree',self.points(height=3.5,base=1.),self.mask())[0],'pine')
        self.assertEqual(P.automatic_kind('pine',self.points(),self.mask())[0],'pine')
        self.assertEqual(P.automatic_kind('plant',self.points(),self.mask(),indoor=True)[0],'plant_pot')

    def test_bush_instance_is_separated_from_a_connected_tree_semantic_region(self):
        tree=np.zeros((200,300),np.uint8);cv2.rectangle(tree,(100,20),(170,80),1,-1);cv2.rectangle(tree,(130,60),(142,170),1,-1)
        bush=self.mask();mask=(tree>0)|bush
        yy,xx=np.indices(mask.shape);z=np.where(yy>110,.5,4.)
        fr=types.SimpleNamespace(X=np.dstack([(xx-150)/50,np.full(mask.shape,6.),z]),valid=np.ones(mask.shape,bool),gp=(0,0,0),vegetation_instances=[bush])
        pieces=P.split_shrubs(fr,mask,'tree')
        self.assertGreaterEqual(len(pieces),2)
        self.assertEqual({P.automatic_kind(n,fr.X[m],m)[0] for m,n in pieces},{'pine','bush'})
        self.assertTrue(all(not (a&b).any() for i,(a,_) in enumerate(pieces) for b,_ in pieces[i+1:]))

    def test_semantic_foliage_keeps_its_silhouette_across_invalid_depth(self):
        mask=self.mask();valid=np.zeros(mask.shape,bool);valid[140:]=True
        fr=types.SimpleNamespace(sem=np.where(mask,1,0),segm=types.SimpleNamespace(names=['wall','tree']),
            person=np.zeros(mask.shape,bool),bars=np.zeros(mask.shape,bool),valid=valid)
        fr.segm.label=lambda sem,m:('tree',1.)
        pieces=E.soft_parts(fr)
        self.assertEqual(len(pieces),1);np.testing.assert_array_equal(pieces[0][0],mask)

    def test_bad_foliage_depth_is_not_mistaken_for_sky(self):
        mask=self.mask();yy,xx=np.indices(mask.shape);valid=yy%5==0
        fr=types.SimpleNamespace(sem=np.where(mask,1,0),segm=types.SimpleNamespace(names=['wall','tree']),
            person=np.zeros(mask.shape,bool),valid=valid,gp=(0.,0.,0.),img=np.full((*mask.shape,3),[40,120,40],np.uint8),
            X=np.dstack([(xx-150)/80,np.full(mask.shape,6.),(180-yy)/80]),
            NW=np.broadcast_to([0.,-1.,0.],(*mask.shape,3)),rng=np.full(mask.shape,6.))
        region=E.region(fr,mask,False)
        self.assertIsNotNone(region);self.assertNotEqual(region['cls'],'sky')
        np.testing.assert_array_equal(region['mask'],mask)

    def test_sam_concepts_survive_coarse_semantic_labels_and_do_not_merge_into_floor(self):
        mask=self.mask();yy,xx=np.indices(mask.shape)
        fr=types.SimpleNamespace(H=200,W=300,sem=np.zeros(mask.shape,int),segm=types.SimpleNamespace(names=['grass']),
            person=np.zeros(mask.shape,bool),bars=np.zeros(mask.shape,bool),valid=np.ones(mask.shape,bool),gp=(0,0,0),
            img=np.full((*mask.shape,3),[40,120,40],np.uint8),X=np.dstack([(xx-150)/80,np.full(mask.shape,6.),(180-yy)/800]),
            NW=np.broadcast_to([0.,-1.,0.],(*mask.shape,3)),rng=np.full(mask.shape,6.),
            scene_concepts=[{'mask':mask,'label':'bush','score':.9},{'mask':mask,'label':'hedge','score':.8}])
        regions,claimed=E.shrub_concepts(fr,False)
        self.assertEqual(len(regions),1);self.assertEqual(regions[0]['label'][0],'bush')
        np.testing.assert_array_equal(claimed,mask)
        floor=copy.deepcopy(regions[0]);floor.pop('detector');floor.update(mask=~mask,area=float((~mask).mean()),cls='floor',
            P=fr.X[~mask],plane=(np.array([0,0,1.]),np.array([0,6,0]),0.),planar=True)
        self.assertEqual(len(E.merge_regions(regions+[floor],200,300)),2)

    def test_concept_results_are_cached_and_invalid_results_requeue_themselves(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=pathlib.Path(tmp)/'sam.pt';checkpoint.write_bytes(b'checkpoint')
            c={'sam3_checkpoint':str(checkpoint),'sam3_python':sys.executable};rgb=np.zeros((20,30,3),np.uint8);pending=[]
            SAM.segment(rgb,c,cache_dir=tmp,_queue=pending)
            self.assertEqual(pending[0][1]['vegetation_prompts'],['bush','shrub','hedge'])
            path=pending[0][3]
            common=dict(masks=np.zeros((0,20,30),bool),box_masks=np.zeros((0,20,30),bool),peak_vram_mb=1.)
            np.savez_compressed(path,**common,concept_masks=np.ones((1,20,30),bool),concept_labels=np.array(['bush']),concept_scores=np.array([.9]))
            result=SAM.segment(rgb,c,cache_dir=tmp)
            self.assertEqual(result['concept_labels'].tolist(),['bush']);self.assertEqual(result['concept_masks'].shape,(1,20,30))
            np.savez_compressed(path,**common,concept_masks=np.ones((2,20,30),bool),concept_labels=np.array(['bush']),concept_scores=np.array([.9]))
            again=[];SAM.segment(rgb,c,cache_dir=tmp,_queue=again)
            self.assertEqual(len(again),1);self.assertEqual(len(list(pathlib.Path(tmp).glob('*.invalid_*'))),1)
            for broken in (b'',b'PK\x03\x04truncated zip file'):
                pathlib.Path(path).write_bytes(broken);again=[]
                SAM.segment(rgb,c,cache_dir=tmp,_queue=again)
                self.assertEqual(len(again),1);self.assertFalse(pathlib.Path(path).exists())

    def test_vegetation_query_setting_only_invalidates_background(self):
        import dependencies as DEP
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);video=root/'clip.mp4';video.write_bytes(b'video')
            for folder in ('work','marks','out'):(root/folder).mkdir()
            args=(str(video),{'character_style':'sample'},root/'work',root/'marks',root/'out')
            before=DEP.inputs(*args)
            after=DEP.inputs(args[0],dict(args[1],sam3_vegetation_prompts=False),*args[2:])
            self.assertEqual(before['fit'],after['fit']);self.assertEqual(before['props'],after['props'])
            self.assertNotEqual(before['environment'],after['environment'])

    def test_exported_fitted_shrub_uses_the_geometry_verified_by_the_fitter(self):
        from build_env import build,Cam
        obj={'name':'shrub','type':'bush','pos':[0,6.,.1],'size':[1.3,.7,.8],'yaw':0,'_fit':{'version':O.VERSION}}
        view=observation(obj);obj['_observations']=[view]
        data={'frames':1,'res':[192,144],'lens':33.75,'sensor':36.,'cam':[[0,0,1.4,90,0,0]],'zoom':[1.]}
        meshes=build({'ground':{'size':0},'objects':[{'name':'offset','type':'box','pos':[5,8,0],'size':[1,1,1]},obj]},Cam(data))
        group=[m for m in meshes if m.get('_observations')]
        self.assertGreater(O.mesh_metrics(group)[0]['iou'],.98)

    def test_bush_dimensions_remain_bounded_for_small_flat_and_wide_prototypes(self):
        for kind in ('bush','hedge'):
            for w,h,d in ((.12,.08,.05),(5.,.4,.2),(1.3,1.,.7)):
                for seed in (0,17,300):
                    parts=props.build(kind,w,h,d,seed);v=np.concatenate([v for _,v,*_ in parts])
                    np.testing.assert_allclose(v.min(0),[-w/2,-d/2,0],atol=1e-6)
                    np.testing.assert_allclose(v.max(0),[w/2,d/2,h],atol=1e-6)
                    self.assertLessEqual(sum(len(f) for _,_,f,_ in parts),1200)

    def test_automatic_fitting_uses_detected_bounds_and_retries_a_poor_shrub_fit(self):
        actual={'type':'hedge','pos':[0,6.,0.],'size':[2.5,.7,.6],'yaw':0}
        view=observation(actual);dims=np.array(actual['size']);points=np.concatenate([v for _,v,_ in O._parts('hedge',dims)])+actual['pos']
        view.update(primary=True,detection_box=O.bounds(points,view).tolist())
        target=O.unbits(view['mask'],view['shape']);target[:,::2]=False;view['mask']=O.bits(target)
        obj=dict(actual,type='bush',_src={'cls':'object'},_observations=[view])
        O.seed_annotation(obj,view);O.fit(obj,max_evaluations=20)
        self.assertTrue(obj['_fit']['detection_anchored']);self.assertFalse(obj['_fit']['annotation_anchored'])
        recovery=obj['_fit']['automatic_recovery']
        self.assertEqual(recovery['attempts'],1)
        self.assertGreaterEqual(recovery['iou_after'],recovery['iou_before'])


if __name__=='__main__':unittest.main()
