"""Create a searchable offline catalog and contact sheets for saved object assets.

Run with the app's Python: python pipeline/build_object_catalog.py
Uses Pillow; the Blender generator writes source thumbnails under output/.
"""
import html
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from object_asset_shapes import GROUPS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/objects'
BASE_GROUPS={
 'Furniture': 'sofa armchair ottoman chair office_chair stool table coffee_table pool_table desk bench bed cabinet wardrobe chest_of_drawers shelf bookshelf counter kitchen_island'.split(),
 'Kitchen and household': 'sink stove oven fridge microwave washer dishwasher bathtub shower bin vase pot cup lamp floor_lamp table_lamp radiator'.split(),
 'Electronics and office': 'tv monitor'.split(),
 'Street and architecture': 'wall door window curtain blind picture mirror pillar pole streetlight traffic_light stairs railing fence awning fireplace booth arcade_machine'.split(),
 'Transport and storage': 'car truck van bus box barrel'.split(),
 'Park and nature': 'tree pine_tree bush'.split(),
}


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    metadata=json.loads((OUT/'objects.json').read_text(encoding='utf-8'))
    groups={name:group for group,names in BASE_GROUPS.items() for name in names}
    groups.update({name:group for group,names in GROUPS.items() for name in names})
    thumbs=OUT/'previews';thumbs.mkdir(exist_ok=True)
    items=sorted(manifest,key=lambda n:(groups.get(n,'Other'),n))
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    sheets=[]
    for page in range(math.ceil(len(items)/64)):
        subset=items[page*64:(page+1)*64];w,h,cols=192,211,8
        image=Image.new('RGB',(cols*w,math.ceil(len(subset)/cols)*h),'#e9ecef');draw=ImageDraw.Draw(image)
        for i,name in enumerate(subset):
            source=ROOT/'output/object_asset_previews'/(name+'.png')
            assert source.exists(), 'Missing generated preview: '+name
            shutil.copyfile(source,thumbs/source.name)
            x=i%cols*w+4;y=i//cols*h+4
            with Image.open(source) as thumb:image.paste(thumb.convert('RGB').resize((184,184)),(x,y))
            draw.text((x+1,y+185),name,font=font,fill='#17212b')
        target='catalog.jpg' if page==0 else 'catalog_%02d.jpg'%(page+1)
        image.save(OUT/target,quality=90);sheets.append(target)
    cards=[]
    for name in items:
        info=manifest[name];group=groups.get(name,'Other');label=name.replace('_',' ')
        size=' × '.join('%g'%v for v in info['size_m'])
        placement='Ground' if metadata[name]['on_floor'] else 'Observed height'
        cards.append('<a class="asset" href="%s.blend" data-group="%s" data-name="%s"><img loading="lazy" width="192" height="192" src="previews/%s.png" alt="%s"><strong>%s</strong><span>%s</span><small>%s m · %s faces<br>%s</small></a>'%(name,html.escape(group),html.escape(label),name,html.escape(label),html.escape(label),html.escape(group),size,info['faces'],placement))
    options=''.join('<option>%s</option>'%html.escape(g) for g in sorted(set(groups.get(n,'Other') for n in items)))
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Background object library</title>
<style>body{font:16px system-ui;background:#eff1f3;color:#202a32;margin:24px}h1{margin-bottom:6px}header{position:sticky;top:0;background:#eff1f3;padding:12px 0;z-index:1}input,select{padding:10px;border:1px solid #abb6bf;border-radius:6px;font:inherit;margin:4px}input{width:min(380px,70vw)}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}.asset{display:flex;flex-direction:column;background:white;border:1px solid #ccd2d7;border-radius:8px;overflow:hidden;color:inherit;text-decoration:none;padding-bottom:12px}.asset[hidden]{display:none}.asset img{width:100%;height:auto;aspect-ratio:1}.asset strong,.asset span,.asset small{margin:7px 12px 0}.asset span,.asset small{color:#586875;font-size:12px}.asset:hover{border-color:#177b85;box-shadow:0 3px 12px #0002}#count{margin:8px}</style>
<h1>Background object library</h1><p>Simple editable Blender placeholders. Metre units, front −Y, base at zero. Select an object to open or download its .blend file.</p><header><input id="search" placeholder="Search objects" aria-label="Search objects"><select id="category" aria-label="Category"><option value="">All categories</option>OPTIONS</select><span id="count"></span></header><main id="grid">CARDS</main>
<script>const cards=[...document.querySelectorAll('.asset')],search=document.getElementById('search'),category=document.getElementById('category');function filter(){let count=0;const query=search.value.trim().toLowerCase();for(const card of cards){card.hidden=!!((query&&!card.dataset.name.includes(query))||(category.value&&card.dataset.group!==category.value));if(!card.hidden)count++}document.getElementById('count').textContent=`${count} / ${cards.length} objects`}search.oninput=filter;category.onchange=filter;filter();</script></html>'''
    (OUT/'catalog.html').write_text(document.replace('OPTIONS',options).replace('CARDS','\n'.join(cards)),encoding='utf-8')
    (OUT/'catalog_index.json').write_text(json.dumps({'count':len(items),'sheets':sheets,'categories':{group:sum(groups.get(n,'Other')==group for n in items) for group in sorted(set(groups.values()))}},indent=2)+'\n',encoding='utf-8')
    print('Catalog ready:',len(items),'objects,',len(sheets),'contact sheets')


if __name__=='__main__':main()
