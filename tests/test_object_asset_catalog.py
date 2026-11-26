"""Contract checks for the shipped placeholder library and its offline browser."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
import assets_lib
from object_asset_shapes import EXTRA_KINDS


class ObjectAssetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'assets/objects'
        self.manifest=json.loads((self.folder/'manifest.json').read_text(encoding='utf-8'))
        self.metadata=json.loads((self.folder/'objects.json').read_text(encoding='utf-8'))

    def test_every_library_asset_has_geometry_metadata_and_preview(self):
        catalog=assets_lib.catalog({'object_assets':str(self.folder)})
        self.assertEqual(set(catalog),set(self.manifest))
        self.assertTrue(set(EXTRA_KINDS)<=set(catalog))
        self.assertGreaterEqual(len(catalog),390)
        for name,info in self.manifest.items():
            with self.subTest(name=name):
                self.assertTrue((self.folder/info['file']).is_file())
                self.assertTrue((self.folder/'previews'/(name+'.png')).is_file())
                self.assertTrue(all(v>0 for v in info['size_m']))
                self.assertLess(info['faces'],3000)
                self.assertTrue(self.metadata[name]['prompt'])

    def test_support_heights_for_small_and_mounted_objects(self):
        for name in ('smartphone','laptop','kettle','plate','cup','security_camera','power_outlet','whiteboard'):
            self.assertFalse(self.metadata[name]['on_floor'],name)
        for name in ('bicycle','sofa','workbench','fire_hydrant','gazebo','hospital_bed'):
            self.assertTrue(self.metadata[name]['on_floor'],name)

    def test_catalog_pages_cover_the_whole_library(self):
        index=json.loads((self.folder/'catalog_index.json').read_text(encoding='utf-8'))
        self.assertEqual(index['count'],len(self.manifest))
        self.assertEqual(sum(index['categories'].values()),len(self.manifest))
        for sheet in index['sheets']:self.assertTrue((self.folder/sheet).is_file())
        page=(self.folder/'catalog.html').read_text(encoding='utf-8')
        for name in self.manifest:self.assertIn('href="'+name+'.blend"',page)

    @unittest.skipUnless(shutil.which('node'),'Node required to exercise offline catalog filtering')
    def test_search_and_category_filters_use_actual_browser_script(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync('assets/objects/catalog.html','utf8');
const cards=[{dataset:{name:'kettle',group:'Kitchen and household'}},{dataset:{name:'bicycle',group:'Transport and storage'}},{dataset:{name:'motorcycle',group:'Transport and storage'}}];
const fields={search:{value:''},category:{value:''},count:{}};
const context=vm.createContext({document:{querySelectorAll:()=>cards,getElementById:id=>fields[id]}});
vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1],context);
assert.equal(fields.count.textContent,'3 / 3 objects');
fields.search.value='KETTLE';fields.search.oninput();assert(!cards[0].hidden&&cards[1].hidden);
fields.search.value='';fields.category.value='Transport and storage';fields.category.onchange();assert(cards[0].hidden&&!cards[1].hidden&&!cards[2].hidden);
fields.search.value='bicycle';fields.search.oninput();assert(!cards[1].hidden&&cards[2].hidden);
fields.search.value='missing';fields.search.oninput();assert.equal(fields.count.textContent,'0 / 3 objects');
'''
        result=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
