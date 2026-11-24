# -*- coding: utf-8 -*-
"""The simple flow: source folder -> package names -> prepare -> review -> Finish (skipped when nothing changed).
   .venv\\Scripts\\python -m unittest discover -s tests -v"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from agent import config, server  # noqa: E402
from agent import clips as clipsmod  # noqa: E402
import package as pkg  # noqa: E402  (pipeline)
import run_pipeline as rp  # noqa: E402
from test_agent import make_video  # noqa: E402


class TestPackageNames(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(pkg.title_slug("The Hidden Blade!"), "the_hidden_blade")
        self.assertEqual(pkg.title_slug("  \u9690\u5251  "), "")

    def test_numbering_continues_and_keeps_existing(self):
        names = {"a": "the_hidden_blade-1", "b": "the_hidden_blade-7", "x": "other-3"}
        plan = pkg.propose_names(["a", "c", "d"], "The hidden blade", names)
        self.assertEqual(plan["a"], ("the_hidden_blade-1", False))
        self.assertEqual(plan["c"], ("the_hidden_blade-8", True))
        self.assertEqual(plan["d"], ("the_hidden_blade-9", True))
        self.assertTrue(all(pkg.PKG_RE.match(v[0]) for v in plan.values()))

    def test_no_title_no_name(self):
        self.assertEqual(pkg.propose_names(["c"], "", {}), {"c": (None, False)})

    def test_assign_writes_only_new(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "names.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"a": "film-1"}, fh)
            with mock.patch.object(pkg, "NAMES_P", p):
                new = pkg.assign_names(["a", "b"], "film")
                self.assertEqual(new, {"b": "film-2"})
                self.assertEqual(pkg.package_names(), {"a": "film-1", "b": "film-2"})
                self.assertEqual(pkg.assign_names(["a", "b"], "film"), {})      # never renames
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestMemberNames(unittest.TestCase):
    def test_members_keep_their_own_names_and_never_share_a_number(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "names.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"x": "film-1"}, fh)                                  # from before members: stays
            anna, ben = {"_user": "anna"}, {"_user": "ben"}
            with mock.patch.object(pkg, "NAMES_P", p):
                self.assertEqual(pkg.assign_names(["a", "b"], "film", anna), {"a": "film-2", "b": "film-3"})
                self.assertEqual(pkg.assign_names(["a"], "film", ben), {"a": "film-4"}, "Ben's clip a is not Anna's")
                self.assertEqual(pkg.package_names(anna), {"a": "film-2", "b": "film-3"})
                self.assertEqual(pkg.package_name("a", ben), "film-4")
                self.assertEqual(pkg.package_names(), {"x": "film-1"})
                with mock.patch.dict(os.environ, {"BA_USER": "ben"}):          # inside a job run for Ben
                    self.assertEqual(pkg.package_name("a"), "film-4")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_member_folders(self):
        import common
        c = common.member_cfg({"work_root": r"D:\w", "submit_root": r"D:\s", "source_root": r"D:\SOURCE"}, "anna", r"E:\anna")
        self.assertEqual((c["source_root"], c["work_root"], c["submit_root"]), (r"E:\anna", r"D:\w\_users\anna", r"D:\s\_users\anna"))
        self.assertEqual(c["marks_root"], os.path.join(rp.AGENT, "output", "_users", "anna"))
        self.assertEqual(common.output_dir(c), os.path.join(rp.AGENT, "output", "_users", "anna"))
        with self.assertRaises(ValueError):
            common.member_cfg(c, "ben", "")


class TestBatchedDetector(unittest.TestCase):
    """the batched person detector's own NMS gives what the original model's built-in one + rtmlib gave"""
    def test_nms_per_class_top_and_threshold(self):
        import det_batch_model as dbm
        import numpy as np
        boxes = np.array([[0, 0, 100, 200], [2, 2, 102, 202], [300, 0, 400, 200], [0, 0, 100, 200], [500, 0, 600, 90]], np.float32)
        scores = np.zeros((5, 80), np.float32)
        scores[0, 0], scores[1, 0], scores[2, 0] = 0.9, 0.8, 0.6   # 1 overlaps 0 (IoU 0.94): suppressed; 2 is apart: kept
        scores[3, 5] = 0.7                                       # the same place, another class: NMS is per class, kept
        scores[4, 0] = 0.25                                      # below rtmlib's 0.3: left out
        out = dbm.person_dets(boxes, scores)
        self.assertEqual(out.tolist(), [[0, 0, 100, 200], [0, 0, 100, 200], [300, 0, 400, 200]], "best first")
        self.assertEqual(dbm.person_dets(boxes, np.zeros((5, 80), np.float32)).shape, (0, 4))

    def test_iou_at_the_threshold_is_kept(self):
        import det_batch_model as dbm
        import numpy as np
        b = np.array([[0, 0, 100, 100], [0, 35, 100, 135]], np.float32)       # IoU 65/135 = 0.48: kept
        self.assertEqual(len(dbm.nms(b, np.array([0.9, 0.8]), 0.65)), 2)
        b = np.array([[0, 0, 100, 100], [0, 10, 100, 110]], np.float32)       # IoU 90/110 = 0.82: suppressed
        self.assertEqual(dbm.nms(b, np.array([0.9, 0.8]), 0.65).tolist(), [0])


class TestMarksFolder(unittest.TestCase):
    def test_default_source_unchanged(self):
        c = {"source_root": os.path.join(rp.AGENT, "SOURCE")}
        self.assertEqual(rp.marks_base(c), os.path.join(rp.AGENT, "output"))
        self.assertEqual(rp.marks_dir(os.path.join(rp.AGENT, "SOURCE", "x.mp4"), c), os.path.join(rp.AGENT, "output", "x"))

    def test_other_folders_do_not_share_joints(self):
        a = {"source_root": r"D:\shoot\day1"}; b = {"source_root": r"E:\copy\day1"}
        da, db = rp.marks_dir(r"D:\shoot\day1\x.mp4", a), rp.marks_dir(r"E:\copy\day1\x.mp4", b)
        self.assertNotEqual(da, db)
        self.assertIn(os.path.join("output", "_sources", "day1_"), da)


class TestClipIds(unittest.TestCase):
    def test_same_file_name_other_folder_other_id(self):
        d = tempfile.mkdtemp()
        try:
            for sub in ("a", "b"):
                os.makedirs(os.path.join(d, sub)); make_video(os.path.join(d, sub, "x.mp4"), n=5)
            ids = [clipsmod.Library(dict(config.load(), source_root=os.path.join(d, sub), work_root=os.path.join(d, "w"))).clips[0].id
                   for sub in ("a", "b")]
            self.assertNotEqual(ids[0], ids[1])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestRemoveFromList(unittest.TestCase):
    def test_remove_and_put_back_keeps_the_file_and_the_names(self):
        d = tempfile.mkdtemp(); data = config.DATA
        try:
            config.DATA = os.path.join(d, "data")
            src = os.path.join(d, "src", "FilmA"); os.makedirs(src)
            for n in ("a.mp4", "b.mp4"):
                make_video(os.path.join(src, n), n=5)
            lib = clipsmod.Library(dict(config.load(), source_root=os.path.join(d, "src"), work_root=os.path.join(d, "w")))
            names = sorted(cl.name for cl in lib.clips)
            a = next(cl for cl in lib.clips if cl.video.endswith("a.mp4"))
            lib.remove(a)
            self.assertEqual([cl.video for cl in lib.clips], [os.path.join(src, "b.mp4")])
            self.assertTrue(os.path.exists(a.video), "the file stays where it is")
            self.assertEqual([r["path"] for r in lib.removed], [a.video])
            lib.restore(a.video)
            self.assertEqual(sorted(cl.name for cl in lib.clips), names)
            self.assertEqual(lib.removed, [])
        finally:
            config.DATA = data
            shutil.rmtree(d, ignore_errors=True)


class TestTaskZips(unittest.TestCase):
    """the client's task zips (v2.1 layout, the source at target/original.mp4): the video is written next to the zip
    under the zip's name, listed as a clip, and the zip name is its delivery name"""
    def setUp(self):
        import zipfile
        self.d = tempfile.mkdtemp(); self.data = config.DATA
        config.DATA = os.path.join(self.d, "data"); os.makedirs(config.DATA)
        self.src = os.path.join(self.d, "src"); os.makedirs(self.src)
        v = os.path.join(self.d, "v.mp4"); make_video(v, n=5)
        with open(v, "rb") as fh:
            self.vid = fh.read()
        self.zp = os.path.join(self.src, "p867bed92_t022-1.zip")
        task = {"taskId": "p-1-T022", "taskCode": "T022", "taskName": "11\u670821\u65e5 (3)(21).mp4", "projectTitle": "\u4fa0\u63a2\u6770\u514b",
                "sourceVideo": {"name": "11\u670821\u65e5 (3)(21).mp4", "durationSec": 5.08}}
        with zipfile.ZipFile(self.zp, "w") as z:
            for n in ("comparison/", "target/", "white_model/", "model_3d/", "meta/"):
                z.writestr(n, b"")
            z.writestr("target/original.mp4", self.vid)
            z.writestr("model_3d/p867bed92_t022-1.blend", b"BLENDER")
            z.writestr("meta/task.json", json.dumps(task, ensure_ascii=False).encode("utf-8"))
        done = os.path.join(self.src, "done-1.zip")                      # a finished submission is not a task
        with zipfile.ZipFile(done, "w") as z:
            z.writestr("target/original.mp4", self.vid); z.writestr("white_model/blendervid.mp4", self.vid)
        self.names = os.path.join(self.d, "names.json")
        self.c = dict(config.load(), source_root=self.src, work_root=os.path.join(self.d, "w"), submit_root=os.path.join(self.d, "s"))

    def tearDown(self):
        config.DATA = self.data
        shutil.rmtree(self.d, ignore_errors=True)

    def test_video_comes_out_under_the_zip_name_with_its_delivery_name(self):
        with mock.patch.object(pkg, "NAMES_P", self.names):
            lib = clipsmod.Library(self.c)
            self.assertEqual([cl.rel for cl in lib.clips], ["p867bed92_t022-1.mp4"])
            out = os.path.join(self.src, "p867bed92_t022-1.mp4")
            with open(out, "rb") as fh:
                self.assertEqual(fh.read(), self.vid, "byte for byte")
            cl = lib.clips[0]
            self.assertEqual(cl.to_dict()["zip"], "p867bed92_t022-1.zip")
            self.assertEqual(cl.task["code"], "T022"); self.assertEqual(cl.task["project"], "\u4fa0\u63a2\u6770\u514b")
            self.assertEqual(pkg.package_name(cl.name, self.c), "p867bed92_t022-1")
            m = os.path.getmtime(out)
            lib.rescan()                                                    # not written again
            self.assertEqual(os.path.getmtime(out), m)
            self.assertEqual(len(lib.clips), 1)
            self.assertEqual(lib.zip_problems, [])
            self.assertFalse([f for f in os.listdir(self.src) if f.endswith(".part")])

    def test_a_video_of_that_name_that_is_not_from_the_zip_is_left_alone(self):
        out = os.path.join(self.src, "p867bed92_t022-1.mp4")
        with open(out, "wb") as fh:
            fh.write(b"mine")
        with mock.patch.object(pkg, "NAMES_P", self.names):
            lib = clipsmod.Library(self.c)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"mine")
        self.assertEqual(len(lib.zip_problems), 1)

    def test_preview_lists_without_writing(self):
        with mock.patch.object(pkg, "NAMES_P", self.names):
            lib = clipsmod.Library(self.c, preview=True)
            self.assertEqual([cl.rel for cl in lib.clips], ["p867bed92_t022-1.mp4"])
            self.assertFalse(os.path.exists(os.path.join(self.src, "p867bed92_t022-1.mp4")))
            self.assertEqual(pkg.package_names(self.c), {})
            scan = server.folder_scan(self.c, self.src)
            self.assertEqual([(r["file"], r["package"]) for r in scan["videos"]], [("p867bed92_t022-1.mp4", "p867bed92_t022-1")])
            self.assertEqual(scan["unnamed"], 0)


class TestHeldObjects(unittest.TestCase):
    """objects people carry: which hand, how often seen, and placed on the fitted hand where the picture shows them"""
    @staticmethod
    def kp(wl=(560, 600), wr=(440, 600)):
        k = np.zeros((17, 3)); k[:, 2] = 0.9
        k[:, 0] = 500; k[:, 1] = np.linspace(300, 900, 17)
        k[5, :2], k[6, :2], k[11, :2], k[12, :2] = (540, 420), (460, 420), (530, 640), (470, 640)
        k[9, :2], k[10, :2] = wl, wr
        return k

    def test_attach_hand_and_back(self):
        import held_objects as HO
        people = [(1, self.kp())]
        self.assertEqual(HO.attach([550, 610, 620, 760], people, "held"), (1, "wrist_L"), "a bag hanging below the left hand")
        self.assertIsNone(HO.attach([100, 100, 160, 160], people, "held"), "far from both hands")
        self.assertIsNone(HO.attach([435, 290, 565, 910], people, "held"), "a box that is the person themself")
        other = self.kp(); other[:, 0] += 100                           # a second person beside: their box is no object
        self.assertIsNone(HO.attach([535, 290, 665, 910], [(1, self.kp()), (2, other)], "held"))
        self.assertEqual(HO.attach([465, 430, 535, 630], people, "worn"), (1, "chest"))

    def test_groups_need_enough_frames(self):
        import held_objects as HO
        d = lambda f, label="bag": {"f": f, "box": [0, 0, 10, 10], "score": 0.6, "track": 1, "joint": "wrist_L", "kind": "held", "label": label}
        objs = HO.groups([d(0), d(2), d(4, "box"), d(6)], {1: 10})
        self.assertEqual(len(objs), 1)
        self.assertEqual((objs[0]["label"], objs[0]["shape"], sorted(objs[0]["boxes"])), ("bag", "bag", ["0", "2", "4", "6"]))
        self.assertEqual(HO.groups([d(0), d(2)], {1: 10}), [], "seen twice: a passing false detection")
        self.assertEqual(HO.groups([d(0), d(2), d(4)], {1: 40}), [], "in under a quarter of the frames")

    def test_fill_holds_and_interpolates(self):
        import held_objects as HO
        F = HO.fill({2: [1.0], 6: [3.0]}, 9)
        self.assertEqual(F[:, 0].tolist(), [1, 1, 1, 1.5, 2, 2.5, 3, 3, 3])

    def test_placed_on_the_hand_where_the_picture_shows_it(self):
        import held_objects as HO
        import build_env as BE
        from rig import Rig, JOINTS
        N = 5
        ch = {"height": 1.75, "style": "sample", "track_id": 1, "visible": [True] * N, "root": [[0.0, 5.0, 0.0, 0.0]] * N,
              "pelvis_z": [0.94] * N, "rot": {j: [[0, 0, 0]] * N for j in JOINTS}}
        data = {"res": [1920, 1080], "lens": 35.0, "sensor": 36.0, "frames": N, "cam": [[0.0, 0.0, 1.2, 90.0, 0.0, 0.0]] * N,
                "chars": {"P1": ch}, "props": [{"name": "chair", "keys": []}]}
        cam = BE.Cam(data)
        W = Rig(1.75, "sample").fk((0, 5, 0), 0.0, {}, 0.94)
        u, v = cam.project(W["wrist_L"][1], 0)[0][0]
        box = [u - 40, v + 10, u + 40, v + 150]                    # a bag hanging under the hand
        held = {"step": 1, "objects": [{"track": 1, "joint": "wrist_L", "kind": "held", "label": "bag", "shape": "bag",
                                        "boxes": {str(f): box for f in range(N)}}]}
        made = HO.place(data, held)
        self.assertEqual([(m[1], m[2]) for m in made], [("bag", N)])
        p = data["props"][-1]
        self.assertTrue(p["held"]); self.assertEqual(data["props"][0]["name"], "chair", "other props stay")
        c = cam.project(np.array(p["keys"][2][:3]), 2)[0][0]
        self.assertLess(abs(c[0] - u), 2.0); self.assertLess(abs(c[1] - (v + 80)), 2.0)
        t = 5.0                                                   # the hand is 5 m away: 140 px tall = 140 * 5 / f metres
        self.assertAlmostEqual(np.ptp(np.array(p["v"])[:, 2]), 140 * t / cam.f, delta=0.03)
        HO.place(data, held)
        self.assertEqual(sum(1 for q in data["props"] if q.get("held")), 1, "placing again replaces, never doubles")

    def test_hidden_when_the_person_is_not_seen(self):
        import held_objects as HO
        from rig import JOINTS
        N = 4
        ch = {"height": 1.75, "style": "sample", "track_id": 1, "visible": [True, True, False, False], "root": [[0.0, 5.0, 0.0, 0.0]] * N,
              "pelvis_z": [0.94] * N, "rot": {j: [[0, 0, 0]] * N for j in JOINTS}}
        data = {"res": [1920, 1080], "lens": 35.0, "sensor": 36.0, "frames": N, "cam": [[0.0, 0.0, 1.2, 90.0, 0.0, 0.0]] * N,
                "chars": {"P1": ch}}
        held = {"step": 1, "objects": [{"track": 1, "joint": "chest", "kind": "worn", "label": "backpack", "shape": "bag", "boxes": {"0": [900, 400, 1000, 600]}}]}
        HO.place(data, held)
        self.assertEqual([k[6] for k in data["props"][0]["keys"]], [1.0, 1.0, 0.0, 0.0])


class TestMeasuredSet(unittest.TestCase):
    """the set: pieces named by the segmentation model, soft things built from their measured surface"""
    def test_relief_is_a_closed_solid_on_the_measured_surface(self):
        import relief
        H, W = 60, 80
        ys, xs = np.mgrid[0:H, 0:W]
        X = np.dstack([(xs - 40) * 0.02, np.full((H, W), 4.0) + 0.002 * xs, (H - ys) * 0.02])   # a sheet 4 m in front
        ok = (xs - 40) ** 2 + (ys - 30) ** 2 < 20 ** 2                                         # a round piece of it
        V, F = relief.relief_mesh(X, ok, (0.0, 0.0, 0.6), 0.3)
        V = np.array(V); n = len(V) // 2
        self.assertTrue(np.all(np.abs(V[:n, 1] - 4.04) < 0.1), "the front lies on the measured points")
        self.assertTrue(np.all(V[n:, 1] > V[:n, 1] + 0.25), "the back is pushed away from the camera")
        self.assertLess(np.ptp(V[:n, 0]), 0.85); self.assertGreater(np.ptp(V[:n, 0]), 0.7)   # the outline's width (0.8 m)
        edges = {}
        for f in F:
            for a, b in zip(f, f[1:] + f[:1]):
                k = (min(a, b), max(a, b)); edges[k] = edges.get(k, 0) + 1
        self.assertTrue(all(c == 2 for c in edges.values()), "closed: every edge has two faces")
        self.assertIsNone(relief.relief_mesh(X, np.zeros((H, W), bool), (0, 0, 0.6), 0.3))

    def test_segmentation_class_decides_with_the_geometry(self):
        import env_auto as EA
        R = {"cls": "object", "z": (0.0, 0.5, 1.2), "rng": 3.0, "area": 0.01, "planar": False, "plane": (np.array([0, 1.0, 0]), None, 0)}
        self.assertEqual(EA.sem_decide(R, "plant", 0.9, True), ("object", None, True), "a plant: its measured shape")
        self.assertEqual(EA.sem_decide(R, "curtain", 0.7, True)[2], True)
        self.assertEqual(EA.sem_decide(R, "swivel chair", 0.9, True), ("object", "office_chair", False))
        self.assertEqual(EA.sem_decide(R, "person", 0.9, True)[0], "sky", "a missed person is no set piece")
        self.assertEqual(EA.sem_decide(R, "wall", 0.95, True)[0], "sky", "a scrap of wall the wall piece has")
        self.assertEqual(EA.sem_decide(R, "plant", 0.3, True), ("object", None, False), "unsure: the geometry stays")
        big = dict(R, cls="wall", area=0.2, planar=True, plane=(np.array([0, 1.0, 0]), None, 0))
        self.assertEqual(EA.sem_decide(big, "cabinet", 0.96, True), ("wall", None, False), "a large wall stays a wall")
        tree = dict(R, cls="vegetation")
        self.assertEqual(EA.sem_decide(tree, "tree", 0.9, False), ("vegetation", None, False), "outdoors: the tree model")
        self.assertTrue(EA.sem_decide(tree, "tree", 0.9, True)[2], "indoors: its measured shape")

    def test_every_role_is_known(self):
        import semseg, props
        geo = {"floor", "ground", "ceiling", "sky", "person", "wall", "building", "far", "rock", "tree", "relief"}
        bad = sorted(k for k, v in semseg.ROLE.items() if v not in geo and props.norm_type(v) == "other" and v != "box")
        self.assertEqual(bad, [], "props types that do not exist")

    def test_build_env_builds_a_mesh_piece(self):
        import build_env as BE
        env = {"ground": {"size": 0}, "objects": [{"name": "plant", "type": "mesh", "v": [[0, 3, 0], [1, 3, 0], [0, 3, 1]],
                                                   "f": [[0, 1, 2]], "colour": "tree"}]}
        data = {"res": [1920, 1080], "lens": 35.0, "sensor": 36.0, "cam": [[0.0, 0.0, 1.2, 90.0, 0.0, 0.0]]}
        m = BE.build(env, BE.Cam(data), 0, dx=200.0, tag="_s2")
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["v"][1], [201.0, 3.0, 0.0], "moved with its shot"); self.assertEqual(m[0]["colour"], "tree")


class TestAnnotations(unittest.TestCase):
    """the clip page's object notes: the description decides how each one is built"""
    def test_camera_zoom(self):
        """a zooming lens: the world camera and the scene camera agree, and a point at the frame-0 edge leaves the picture"""
        from common import Camera
        from ground_solve import World
        import build_env as BE
        cam = Camera(1920, 1080, 30.0)
        N = 3; zoom = [1.0, 1.5, 2.0]
        w = World(cam, [np.eye(3)] * N, np.array([0, -1.0, 0]), 1.2, zoom=zoom)
        P = w.on_ground(0, 40.0, 1000.0)                                 # near the left edge at frame 0
        self.assertLess(abs(w.project(P, 0)[0][0, 0] - 40.0), 0.5)
        self.assertLess(w.project(P, 2)[0][0, 0], 0)                     # zoomed x2: out of the picture
        data = {"res": [1920, 1080], "lens": 30.0, "sensor": 36.0, "cam": [w.blender_cam(i) for i in range(N)],
                "zoom": [w.f_at(i) / cam.f for i in range(N)]}
        bc = BE.Cam(data)
        for i in range(N):
            self.assertLess(np.abs(bc.project(P, i)[0] - w.project(P, i)[0]).max(), 0.5)
            C, d = bc.ray(700.0, 900.0, i)
            self.assertLess(np.abs(d - w.ray(i, 700.0, 900.0)).max(), 1e-6)

    def test_distance_levels(self):
        import env_auto as EA
        C = [0.0, 0.0, 1.5]
        ob = lambda what, lvl, near, typ="wall": {"what": what, "type": typ, "dist": near, "_level": lvl, "_C": C,
                                                  "_reach": [near, near + 0.5], "_az": [80.0, 100.0]}
        wall, table, free = ob("back wall", 3, 2.0), ob("table", 1, 3.0), ob("lamp", None, 1.0)
        mesh = dict(ob("plant", 2, 1.0), type="mesh", v=[[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        person = [np.array([[0.0, 6.0, 1.0], [0.2, 6.2, 1.2]])]            # walks 6 m out, straight ahead
        aside = [np.array([[9.0, 0.5, 1.0]])]                              # far off to the side: not in the wall's view
        EA.by_level([wall, table, free, mesh], person + aside)
        self.assertAlmostEqual(table["dist"], 3.0)                         # level 1: as measured
        self.assertAlmostEqual(mesh["_reach"][0], 3.8, places=2)           # level 2: behind the table (3.5 + 0.3)
        self.assertAlmostEqual(mesh["v"][0][1], 3.8, places=2)             # its points pushed along the view
        self.assertGreaterEqual(wall["dist"], 6.2 + 0.4 - 0.01)            # the back: behind the person
        self.assertLess(wall["dist"], 8.0)                                 # (the one to the side does not count)
        self.assertEqual(free["dist"], 1.0)
        self.assertEqual(server.distance_level("4"), 4)
        self.assertIsNone(server.distance_level(""))
        self.assertIsNone(server.distance_level(12))

    def test_description_decides(self):
        import describe as D
        p = lambda w, t="other": D.parse(w, t)
        self.assertEqual((p("white pillar by the door")["mode"], p("white pillar by the door")["kind"]), ("model", "pillar"))
        self.assertEqual(p("desk lamp with white shade")["kind"], "lamp", "the thing is the last noun")
        self.assertEqual(p("the thing on the shelf")["mode"], "shape", "not the shelf it stands on")
        self.assertEqual(p("a big potted plant next to the wall")["mode"], "shape")
        self.assertEqual(p("old sculpture of a horse")["mode"], "shape")
        h = p("plastic bag held in his left hand")
        self.assertEqual((h["mode"], h["side"], h["label"]), ("held", "L", "bag"))
        self.assertTrue(p("backpack")["worn"])
        self.assertEqual(p("phone in her right hand")["label"], "phone")
        self.assertEqual(p("pillar", "door")["kind"], "door", "a type picked in the list wins")
        self.assertEqual(p("distant door")["role"], "far_i")
        self.assertEqual(p("special thing")["role"], "special")


class TestColourCard(unittest.TestCase):
    def test_card_values(self):
        import colour_card as CC
        self.assertEqual(CC.CARD["red"], [214, 62, 61]); self.assertEqual(CC.CARD["bg"], [184, 184, 184])
        self.assertEqual(CC.CARD["near"], [223, 135, 47]); self.assertEqual(CC.CARD["special"], [122, 78, 194])

    def test_roles(self):
        import colour_card as CC
        from rig import JOINTS
        N = 3
        ch = {"height": 1.75, "style": "sample", "track_id": 1, "visible": [True] * N, "root": [[0.0, 5.0, 0.0, 0.0]] * N,
              "pelvis_z": [0.94] * N, "rot": {j: [[0, 0, 0]] * N for j in JOINTS}, "colour": "purple"}
        box = lambda c, s: [[c[0] + sx * s / 2, c[1] + sy * s / 2, c[2] + sz * s / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        mk = lambda obj, v, **kw: dict({"name": obj, "obj": obj, "v": v, "f": [[0, 1, 2]], "colour": "set", "layer": 2}, **kw)
        data = {"res": [1920, 1080], "lens": 35.0, "sensor": 36.0, "frames": N, "cam": [[0.0, 0.0, 1.2, 90.0, 0.0, 0.0]] * N,
                "chars": {"P1": ch}, "palette": {}, "props": [{"name": "held_P1_bag_0", "held": True, "colour": "dark"}],
                "meshes": [mk("wall", box((0, 9, 1.5), 3), cls="wall", kind="patch"),
                           mk("lamp", box((3, 7, 1), 0.4), kind="lamp"),
                           mk("table", box((0, 3.5, 0.9), 0.7), kind="table"),                 # between camera and person
                           mk("door", box((-4, 12, 1), 1), kind="door"),                       # far from him
                           mk("cup", box((0.45, 5.0, 0.95), 0.1), kind="box", what="cup"),     # by his hand
                           mk("special", box((5, 9, 0.5), 1), role="special")]}
        CC.apply(data, log=None)
        col = {m["obj"]: m["colour"] for m in data["meshes"]}
        self.assertEqual(col, {"wall": "bg", "lamp": "lamp", "table": "fg", "door": "far_i", "cup": "near", "special": "special"})
        self.assertEqual(data["props"][0]["colour"], "near", "held: nearby interactive")
        self.assertEqual(data["chars"]["P1"]["colour"], "cyan", "purple is the special colour: a person gets another")
        self.assertEqual(data["palette"]["fg"], [154, 107, 59])


class TestClaudeSet(unittest.TestCase):
    """the Claude set builder's own logic (no API call): the answer it accepts, and measured shapes for plants"""
    def test_parse_answer(self):
        import claude_set as CS
        env = CS.parse_set('A corridor.\n```json\n{"objects": [{"type": "pillar", "bbox": [1, 2, 3, 4], "dist": 3},'
                           '{"type": "box"}, {"what": "no type", "pos": [0, 1, 0]}]}\n```')
        self.assertEqual([o["type"] for o in env["objects"]], ["pillar"], "objects need a type and a placement")
        self.assertIn("ground", env)
        with self.assertRaises(Exception):
            CS.parse_set("no json here")

    def test_spec_leaves_out_people_fitting(self):
        import claude_set as CS
        s = CS.spec_text()
        self.assertIn("## File format", s); self.assertIn("### Types", s)
        self.assertNotIn("People on horses", s); self.assertNotIn("Wrong person tracks", s)

    def test_plants_get_their_measured_shape(self):
        import claude_set as CS
        se = {"objects": [{"type": "plant_pot", "what": "potted fig", "bbox": [300, 400, 700, 1000], "dist": 2.5},
                          {"type": "pillar", "what": "white pillar", "bbox": [0, 0, 200, 1000], "dist": 3}]}
        auto = {"objects": [{"type": "mesh", "name": "object_3", "v": [[0, 0, 0]], "f": [], "_src": {"frame": 0, "px": [320, 420, 690, 990]}},
                            {"type": "mesh", "name": "object_9", "v": [[0, 0, 0]], "f": [], "_src": {"frame": 40, "px": [320, 420, 690, 990]}}]}
        self.assertEqual(CS.measured_shapes(se, auto, 0), 1)
        self.assertEqual([(o["type"], o.get("what")) for o in se["objects"]], [("mesh", "potted fig"), ("pillar", "white pillar")])

    def test_settings_validation(self):
        from agent import config as cfgmod
        d = tempfile.mkdtemp(); p = cfgmod.CFG_PATH
        try:
            cfgmod.CFG_PATH = os.path.join(d, "config.json")
            c = cfgmod.save({"set_builder": "claude", "set_model": "claude-opus-5-5", "set_rounds": 2})
            self.assertEqual((c["set_builder"], c["set_model"]), ("claude", "claude-opus-5-5"))
            for bad in ({"set_builder": "gpt"}, {"set_model": "claude-x"}, {"set_rounds": 9}, {"anthropic_key": "short"}):
                with self.assertRaises(ValueError):
                    cfgmod.save(bad)
            self.assertIn("anthropic_key", cfgmod.SECRETS)
        finally:
            cfgmod.CFG_PATH = p
            shutil.rmtree(d, ignore_errors=True)


class TestFinishFingerprint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10)
        self.c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                      submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"))
        self.cl = clipsmod.Library(self.c).clips[0]
        self.f = clipsmod.folders(self.c, self.cl)
        os.makedirs(self.f["marks"]); os.makedirs(self.f["work"]); os.makedirs(self.c["submit_root"])
        self.write(self.f["review"], {"confirmed": True, "people": {"1": {"role": "main"}}, "saved": "t1"})

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    @staticmethod
    def write(p, d):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh)

    def test_skip_until_something_changes(self):
        c, cl, f = self.c, self.cl, self.f
        fp = clipsmod.fingerprint(c, cl)
        self.write(f["review"], {"confirmed": False, "people": {"1": {"role": "main"}}, "saved": "t2"})
        self.assertEqual(clipsmod.fingerprint(c, cl), fp, "save time and confirm flag are not changes")
        open(f["zip"], "w").close()
        clipsmod.record_finish(c, cl, fp)
        self.write(f["review"], {"confirmed": True, "people": {"1": {"role": "main"}}, "saved": "t3"})
        self.assertTrue(clipsmod.status(c, cl)["done"])
        self.assertEqual(clipsmod.status(c, cl)["label"], "finished")
        self.write(os.path.join(f["marks"], "checked.json"), {"frames": {}, "saved": "t5"})  # the editor saved ...
        clipsmod.record_finish(c, cl, clipsmod.fingerprint(c, cl))                             # ... and was finished
        self.write(os.path.join(f["marks"], "checked.json"), {"frames": {}, "saved": "t6"})  # saved again, nothing new
        self.assertTrue(clipsmod.status(c, cl)["done"], "a save without a new fix is not a change")
        self.write(os.path.join(f["marks"], "checked.json"), {"frames": {"0": []}})     # a joint fix
        st = clipsmod.status(c, cl)
        self.assertFalse(st["done"]); self.assertTrue(st["changed"])
        self.assertEqual(st["label"], "changed since Finish")
        os.remove(os.path.join(f["marks"], "checked.json"))
        self.write(f["review"], {"confirmed": True, "people": {"1": {"role": "ignore"}}, "saved": "t4"})   # a role
        self.assertFalse(clipsmod.status(c, cl)["done"])



class TestJobResult(unittest.TestCase):
    """a job's result is the pipeline's exit code: 0 done, 1 failed (tried again once), 3 review not confirmed (not retried)"""
    def setUp(self):
        from agent import jobs
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10)
        c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                 submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"), max_attempts=2)
        self.store = jobs.JobStore(os.path.join(self.d, "j.db"))
        lib = clipsmod.Library(c)
        self.cl = lib.clips[0]
        app = mock.Mock(c=c, store=self.store)
        app.ws = lambda user: mock.Mock(user=user, c=c, lib=lib)          # the member's workspace
        self.r = jobs.Runner(app)
        self.r.after = lambda *a: None

    def tearDown(self):
        self.store.db.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def run_job(self, code, stage="fit"):
        self.r._cmd = lambda job, cl: [sys.executable, "-c", "print('working'); raise SystemExit(%d)" % code]
        jid = self.store.enqueue(self.cl.id, stage, user="anna")
        self.r._run(self.store.take())
        return jid

    def test_job_runs_in_the_members_folders(self):
        """the job's program gets the member and their folder (BA_USER / BA_SOURCE)"""
        self.r._cmd = lambda job, cl: [sys.executable, "-c", "import os; print('member=' + os.environ['BA_USER'] + '|' + os.environ['BA_SOURCE'])"]
        self.store.enqueue(self.cl.id, "fit", user="anna")
        self.r._run(self.store.take())
        with open(os.path.join(clipsmod.folders(self.r.app.c, self.cl)["work"], "job_fit.log"), encoding="utf-8") as fh:
            self.assertIn("member=anna|" + self.r.app.c["source_root"], fh.read())
        self.store.enqueue(self.cl.id, "fit")                                 # no member: refused, not run as someone
        self.r._run(self.store.take())
        self.assertIn("no member", self.store.latest_for(self.cl.id)["error"])

    def test_one_job_per_clip(self):
        """a forced job queued while its clip's job runs waits for it (both would write scene_data.json)"""
        self.store.enqueue(self.cl.id, "fit", user="anna")
        first = self.store.take()
        self.assertIsNotNone(self.store.enqueue(self.cl.id, "all", force=True, user="anna"))
        self.assertIsNone(self.store.take(), "the clip is busy")
        self.store.finish(first["id"], True)
        self.assertEqual(self.store.take()["stage"], "all")

    def test_exit_codes(self):
        self.run_job(0)                                                     # no "0 failed" line needed any more
        self.assertEqual(self.store.latest_for(self.cl.id)["state"], "done")
        self.run_job(3, "all")
        j = self.store.latest_for(self.cl.id)
        self.assertEqual((j["state"], j["attempts"]), ("failed", 1), "not confirmed: no second attempt")
        self.assertIn("not confirmed", j["error"])
        self.run_job(1)
        self.assertEqual(self.store.latest_for(self.cl.id)["state"], "queued", "a failure is tried once more")
        self.r._run(self.store.take())
        self.assertEqual(self.store.latest_for(self.cl.id)["state"], "failed")
        work = clipsmod.folders(self.r.app.c, self.cl)["work"]
        self.assertTrue(os.path.exists(os.path.join(work, "job_fit.previous.log")), "the first attempt's log is kept")



class TestStartOver(unittest.TestCase):
    """Start over deletes what preparing, review and Finish left for one clip, and nothing else"""
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10); make_video(os.path.join(src, "1-2.mp4"), n=10)
        self.c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                      submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"))
        self.a, self.b = clipsmod.Library(self.c).clips

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def fill(self, cl):
        f = clipsmod.folders(self.c, cl)
        for p in (f["pose"], f["review"], os.path.join(f["marks"], "checked.json"), f["preview"], f["scene"],
                  os.path.join(f["work"], "finished.json"), os.path.join(f["submit"], "meta", "meta.json"), f["zip"]):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write("{}")
        return f

    def test_reset_one_clip(self):
        fa, fb = self.fill(self.a), self.fill(self.b)
        self.assertEqual(clipsmod.status(self.c, self.a)["label"], "finished")
        self.assertEqual(len(clipsmod.prepared_paths(self.c, self.a)), 4)        # marks, work, submit folder, zip
        r = clipsmod.reset(self.c, self.a)
        self.assertEqual((len(r["removed"]), r["failed"]), (4, []))
        self.assertEqual(clipsmod.prepared_paths(self.c, self.a), [])
        self.assertEqual(clipsmod.status(self.c, self.a)["label"], "not prepared")
        self.assertTrue(os.path.exists(self.a.video), "the video stays")
        self.assertTrue(os.path.exists(fb["pose"]) and os.path.exists(fb["zip"]), "the other clip is untouched")
        for root in (self.c["work_root"], self.c["submit_root"], self.c["marks_root"]):
            self.assertTrue(os.path.isdir(root), "the folders themselves stay")

    def test_never_outside_the_app_folders(self):
        self.assertFalse(clipsmod._inside(self.c["work_root"], [self.c["work_root"]]), "a root itself is not inside")
        self.assertFalse(clipsmod._inside(self.a.video, [self.c["work_root"], self.c["submit_root"]]))
        self.assertTrue(clipsmod._inside(os.path.join(self.c["work_root"], "x"), [self.c["work_root"]]))


class TestProgressAndFlags(unittest.TestCase):
    def test_progress_from_log(self):
        d = tempfile.mkdtemp()
        try:
            lp = os.path.join(d, "job_fit.log")
            cl = mock.Mock(); cl.name = "x"
            with mock.patch.object(server, "folders", return_value={"work": d}):
                with open(lp, "w", encoding="utf-8") as fh:
                    fh.write("  detected 40 frames (frame 79/137, people 2)  (139.6s)\n")
                p = server.job_progress({}, cl, {"stage": "fit", "log": lp})
                self.assertTrue(10 < p["pct"] < 20, p); self.assertIn("40 of 137", p["step"])
                with open(lp, "a", encoding="utf-8") as fh:
                    fh.write("  detected 86 frames (frame 40/137, people 2)  (170.2s)\n")
                p2 = server.job_progress({}, cl, {"stage": "fit", "log": lp})
                self.assertGreater(p2["pct"], p["pct"])                       # never steps back
                with open(lp, "a", encoding="utf-8") as fh:
                    fh.write("  marks: x\n  lens: 40.7 mm\n  [shot 1: frames 0-136]\n  camera ...\n")
                p = server.job_progress({}, cl, {"stage": "fit", "log": lp})
                self.assertEqual(p["pct"], 58); self.assertIn("camera", p["step"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_person_flags(self):
        rep = {"quality": {"people": [
            {"track": 3, "flags": ["pose never changes: a statue or a picture?", "jerky (12 sudden movements per 100 frames)"], "jump_frames": [5, 40]},
            {"track": 1, "flags": []}]}, "left_out": [{"track": 7, "why": "short and unclear (0.5 s)"}]}
        f = server.person_flags(rep)
        self.assertIn("statue", f["3"]["not_person"]); self.assertEqual(len(f["3"]["flags"]), 1)
        self.assertIsNone(f["1"]["not_person"])
        self.assertIn("short and unclear", f["7"]["not_person"])
        self.assertEqual(server.jump_frames(rep), {"3": [5, 40]})
        self.assertEqual(server.jump_frames(rep, {"9": 3}), {"3": [5, 40], "9": [5, 40]})   # joined track: old editor id too


class TestSubmissionsPage(unittest.TestCase):
    def test_v21_meta_and_package_name(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "clip_a", "meta"))
            with open(os.path.join(d, "clip_a", "meta", "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"duration": 4.5, "output_video": {"resolution": [1920, 1080]}}, fh)
            with mock.patch.object(pkg, "package_names", return_value={"clip_a": "film-3"}):
                row = server.submissions({"submit_root": d})[0]
            self.assertEqual((row["package"], row["duration"], row["resolution"]), ("film-3", 4.5, [1920, 1080]))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_download_name(self):
        self.assertEqual(server.download_name("the_hidden_blade-3.zip"),
                         "attachment; filename=\"the_hidden_blade-3.zip\"; filename*=UTF-8''the_hidden_blade-3.zip")
        self.assertIn("filename*=UTF-8''%E9%9A%90", server.download_name("\u9690\u5251.zip"))   # non-ASCII kept in the UTF-8 form

    def test_character_file_follows_style(self):
        self.assertEqual(server.char_file({"character_style": "master"}), "human_master.json")
        self.assertEqual(server.char_file({"character_style": "base21"}), "base_character_v21.json")
        self.assertTrue(os.path.exists(os.path.join(ROOT, "assets", server.char_file(config.load()))))


class TestEditorColours(unittest.TestCase):
    def test_role_colours_and_joined_tracks(self):
        rv = {"people": {"1": {"role": "main", "colour": "red"}, "2": {"role": "ignore", "colour": "purple"}}}
        out = server.editor_colours(rv, {"3": 1})
        self.assertEqual(out["1"], {"hex": "#ff4d4d", "role": "Main"})
        self.assertEqual(out["2"]["hex"], server.DRAW_HEX["grey"], "not a person: grey")
        self.assertEqual(out["3"], out["1"], "a joined track keeps the colour of the track it joined")
        self.assertEqual(server.editor_colours(None), {})


class TestCheckPage(unittest.TestCase):
    def test_current_template_used(self):
        d = tempfile.mkdtemp()
        try:
            c = {}; cl = mock.Mock()
            with open(os.path.join(d, "check.html"), "w", encoding="utf-8") as fh:
                fh.write('<html><head></head><script>\nconst PAGE = {"clip": "x", "frames": []};\nconst cv = 1;</script></html>')
            with mock.patch.object(server, "folders", return_value={"marks": d}):
                html = server.check_page(c, cl)
            self.assertIn('const PAGE = {"clip": "x", "frames": []};', html)
            self.assertIn("nextFlagged", html)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
