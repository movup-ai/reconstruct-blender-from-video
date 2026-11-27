# -*- coding: utf-8 -*-
"""The run's object list: Mask2Former's 150 ADE20K names -> the asset (assets\\objects\\<asset>.blend) placed for it.

Mask2Former (models\\mask2former_ade_panoptic) finds and names the objects; this table says which model stands in for
each name.  None: not an object to place - people (the characters), and the ground, sky, water and landscape (the run
builds the floor; the rest is background colour).  A name whose .blend is missing is reported and left out.

  ASSET[ade name]    -> asset name or None
  asset_of(ade name) -> asset name or None
  wanted()           -> every asset name the table uses (the .blend files the run can place)"""

ASSET = {
    # not placed: people and scenery
    "person": None, "sky": None, "floor": None, "ceiling": None, "road": None, "grass": None, "sidewalk": None,
    "earth": None, "mountain": None, "water": None, "sea": None, "field": None, "sand": None, "path": None,
    "runway": None, "river": None, "hill": None, "dirt track": None, "land": None, "lake": None, "waterfall": None,
    "swimming pool": None, "pier": None,
    # the same name
    "wall": "wall", "tree": "tree", "bed": "bed", "cabinet": "cabinet", "door": "door", "table": "table",
    "curtain": "curtain", "chair": "chair", "car": "car", "sofa": "sofa", "shelf": "shelf", "house": "house",
    "mirror": "mirror", "rug": "rug", "armchair": "armchair", "fence": "fence", "desk": "desk", "rock": "rock",
    "wardrobe": "wardrobe", "lamp": "lamp", "bathtub": "bathtub", "railing": "railing", "cushion": "cushion",
    "box": "box", "chest of drawers": "chest_of_drawers", "counter": "counter", "sink": "sink",
    "fireplace": "fireplace", "stairs": "stairs", "pool table": "pool_table", "pillow": "pillow", "blind": "blind",
    "coffee table": "coffee_table", "toilet": "toilet", "flower": "flower", "book": "book", "bench": "bench",
    "stove": "stove", "kitchen island": "kitchen_island", "boat": "boat", "arcade machine": "arcade_machine",
    "bus": "bus", "towel": "towel", "truck": "truck", "awning": "awning", "streetlight": "streetlight",
    "booth": "booth", "pole": "pole", "escalator": "escalator", "ottoman": "ottoman", "bottle": "bottle",
    "van": "van", "fountain": "fountain", "washer": "washer", "stool": "stool", "barrel": "barrel",
    "basket": "basket", "tent": "tent", "oven": "oven", "ball": "ball", "microwave": "microwave", "pot": "pot",
    "bicycle": "bicycle", "dishwasher": "dishwasher", "sculpture": "sculpture", "vase": "vase",
    "traffic light": "traffic_light", "tray": "tray", "fan": "fan", "plate": "plate", "monitor": "monitor",
    "shower": "shower", "radiator": "radiator", "flag": "flag",
    # another name for the same thing
    "windowpane": "window", "painting": "picture", "poster": "picture", "plant": "planter", "seat": "chair",
    "column": "pillar", "signboard": "shop_sign", "trade name": "shop_sign", "refrigerator": "fridge",
    "case": "display_cabinet", "screen door": "door", "stairway": "stairs", "step": "stairs",
    "bookcase": "bookshelf", "countertop": "counter", "bar": "counter", "palm": "palm_tree",
    "computer": "desktop_computer", "swivel chair": "office_chair", "hovel": "shed", "television receiver": "tv",
    "crt screen": "monitor", "bannister": "handrail", "ship": "boat", "canopy": "awning", "bag": "handbag",
    "minibike": "motorcycle", "cradle": "crib", "ashcan": "bin", "bulletin board": "noticeboard",
    "glass": "drinking_glass", "clock": "wall_clock",
    # objects the generated library did not cover: models added by hand
    "building": "building", "skyscraper": "skyscraper", "tower": "tower", "base": "pedestal",
    "grandstand": "grandstand", "bridge": "bridge", "light": "ceiling_light", "chandelier": "chandelier",
    "airplane": "airplane", "apparel": "clothes", "buffet": "sideboard", "stage": "stage",
    "conveyer belt": "conveyor_belt", "plaything": "toy", "food": "food", "tank": "tank", "animal": "animal",
    "screen": "projection_screen", "blanket": "blanket", "hood": "range_hood", "sconce": "wall_lamp",
}


def asset_of(name):
    return ASSET.get(str(name or "").strip())


def wanted():
    return sorted({a for a in ASSET.values() if a})
