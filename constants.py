# # LLM System Prompt for Semantic Group Assignment
# LLM_SEMANTIC_GROUP_SYSTEM_PROMPT = """You are a classifier that assigns ONE incoming indoor object to the BEST matching group by semantics and common-sense indoor context.

# ## INPUT FORMAT
# - A numbered list of groups, each line like:
#   Group <int>: <comma-separated items>
# - An "Incoming object: <string>"

# ## TASK
# - Return ONLY the integer group id (no words, no punctuation, no spaces). If no group fits, return -1.

# ## DECISION RULES (apply in order)
# 1) Exact / synonym match: If the incoming object is the same as, a common synonym of, or a near-lexical variant of any item in a group (e.g., "sofa"≈"couch", "lavatory"≈"toilet"), select that group.
# 2) Hypernym/hyponym match: If the object is a type of an item listed in a group or vice versa (e.g., "armchair"→"chair", "appliances"←"microwave"), select that group.
# 3) Functional/affordance equivalence: If the object is functionally the same as items in a group (e.g., "bookcase"≈"shelf"), select that group.
# 4) Room-type co-occurrence PRIOR: If no direct match by 1–3, choose the group whose items most commonly co-occur with the object in typical indoor settings (e.g., bed↔nightstand; sink↔towel).
# 5) Tie-breakers: prefer (a) Rule 1 over 2 over 3 over 4; if still tied, choose the LOWER group id.
# 6) Unknowns: If none of the above give a reasonable fit, output -1.

# ## NORMALIZATION
# - Lowercase the incoming object and group items.
# - Strip plurals and descriptors (e.g., "wooden dining chairs"→"chair").
# - Map common aliases (e.g., sofa↔couch, tv↔television, dresser↔chest_of_drawers).

# ## OUTPUT
# - Print ONLY the integer group id (e.g., 0). No explanation. No extra tokens.

# ## EXAMPLE
# Input:
# Group 0: chair, table, couch
# Group 1: sink, toilet, bathtub, shower
# Group 2: tv, fireplace, bookshelf
# Incoming object: sofa
# Output:
# 0
# """
# LLM System Prompt for Semantic Group Assignment
LLM_SEMANTIC_GROUP_SYSTEM_PROMPT = """You are a classifier that assigns ONE incoming indoor object to the BEST matching group by semantics and common-sense indoor context.

## INPUT FORMAT
- A numbered list of groups, each line like:
  Group <int>: <comma-separated items>
- An "Incoming object: <string>"

## TASK
- Return ONLY the integer group id (no words, no punctuation, no spaces). If no group fits, return -1.

## DECISION RULES (apply in order)
1) Exact / synonym match: If the incoming object is the same as, a common synonym of, or a near-lexical variant of any item in a group (e.g., "sofa"≈"couch", "lavatory"≈"toilet"), select that group.
2) Hypernym/hyponym match: If the object is a type of an item listed in a group or vice versa (e.g., "armchair"→"chair", "appliances"←"microwave"), select that group.
3) Functional/affordance equivalence: If the object is functionally the same as items in a group (e.g., "bookcase"≈"shelf"), select that group.
4) Room-type co-occurrence PRIOR: If no direct match by 1–3, choose the group whose items most commonly co-occur with the object in typical indoor settings (e.g., bed↔nightstand; sink↔towel).
5) Tie-breakers: prefer (a) Rule 1 over 2 over 3 over 4; if still tied, choose the LOWER group id.
6) Unknowns: If none of the above give a reasonable fit, output -1.

## NORMALIZATION
- Lowercase the incoming object and group items.
- Strip plurals and descriptors (e.g., "wooden dining chairs"→"chair").
- Map common aliases (e.g., sofa↔couch, tv↔television, dresser↔chest_of_drawers).

## OUTPUT
- Print ONLY the integer group id (e.g., 0). No explanation. No extra tokens.

"""

# # LLM System Prompt for Object Selection
# LLM_OBJECT_SELECTION_SYSTEM_PROMPT = """You are an expert in indoor common‑sense spatial reasoning and household object co‑location. Your job is to pick **one** candidate object from a list that is **most likely to be physically near the target object** inside a typical house.

# ### Input format

# * A list of candidates, one per line, formatted like:
#   `object 0: <name>`
#   `object 1: <name>`
#   ...
#   (The word may appear misspelled as “obejct”; treat it the same as “object”.)
# * A final line: `target object: <name>`

# ### What to do (internally—do not explain)

# 1. **Normalize names** (case/typos; map synonyms: fridge↔refrigerator, TV↔television, sofa↔couch, cellphone↔phone, trash can↔bin, etc.).
# 2. **Infer likely room/zone** for each item and the target (kitchen, bathroom, bedroom, living room, office, dining room, laundry, entryway, closet, garage/utility).
# 3. **Score co‑location** using everyday pairings and surfaces/containers:

#    * **Strong functional pairs** (highest):
#      toothbrush↔toothpaste/sink, toilet paper↔toilet, towel↔shower, mug↔kettle/coffeemaker, bowl/spoon↔cereal, pan↔stove, knife↔cutting board, remote↔TV/couch, controller↔console/TV, mouse/keyboard↔computer/desk, charger↔phone/bedside table, detergent↔washer, laundry basket↔washer/dryer, keys↔entry table/door, hanger↔closet/rod, book↔bookshelf, leash↔door/dog bowl, litter scoop↔litter box.
#    * **Same‑room affinity** (high): any candidate typically found in the same room as the target.
#    * **Support surfaces/containers** (medium): nightstand↔bed, TV stand↔TV, desk↔laptop/monitor, sink/cabinet↔toiletries, drawer/rack/shelf that commonly holds the target.
#    * **General household plausibility**: penalize outdoor‑only or mismatched items for an indoor home context.
# 4. **Never select the target itself** if it appears among candidates; choose the best *associated* item.
# 5. **Tie‑breakers (in order):**

#    * Pick the candidate with the strongest direct functional relationship.
#    * If still tied, choose the one most immediately adjacent in typical use (e.g., toothpaste is nearer to toothbrush than to towel).
#    * If still tied, select the object that serves as the **holder/surface** for the target (e.g., nightstand for lamp).
#    * If still tied, pick the lowest `object_id`.

# ### Output format

# * **Reply with only the integer `object_id`** of your chosen candidate.
# * **No JSON, no words, no punctuation, no leading/trailing spaces, no explanations.**

# ### Examples

# Input:
# object 0: remote control
# object 1: pillow
# object 2: kettle
# target object: TV
# Output:
# 0

# Input:
# object 0: nightstand
# object 1: couch
# object 2: dresser
# target object: bed
# Output:
# 0

# """


template="""






"""
# 
## INPUT FORMAT
# - Lines like: "object <int>: <category>"
# - One line: "target object: <category>"

# ## OUTPUT FORMAT
# - Print ONLY the chosen integer object_id.
# - No words, no punctuation, no spaces, no JSON, no trailing characters.

# ## DECISION RULES (in order)
# 1) Alias/Synonym match with the goal’s common companions
   
# 2) Room-type co-occurrence PRIOR (choose the object whose typical ROOM overlaps the goal’s room)
   
# 3) Functional/affordance proximity
   
# 4) Hypernym/hyponym relation
   
# 5) Tie-breakers

# LLM System Prompt for Object Selection
LLM_OBJECT_SELECTION_SYSTEM_PROMPT = """Your job is to pick EXACTLY ONE object_id from a list of indoor objects: the one most likely to be NEAR the goal object in a typical home.

"""

coco_categories = [0, 3, 2, 4, 5, 1]
coco_categories_hm3d2mp3d = [0, 6, 8, 10, 13, 5]
category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
]

category_to_id_gibson = [
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv"
]

category_to_id_mp3d = [
    'chair', #g
    'table', #g
    'picture', #b
    'cabinet', # in resnet
    'cushion', # in resnet
    'sofa', #g
    'bed', #g
    'chest_of_drawers', #b in resnet
    'plant', #g
    'sink', #g
    'toilet', #g
    'stool', #b
    'towel', #b in resnet
    'tv_monitor', #g
    'shower', #b
    'bathtub', #b in resnet
    'counter', #b isn't this table?
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes', # in resnet
    'background'
]

mp3d_category_id = {
    'void': 1,
    'chair': 2,
    'sofa': 3,
    'plant': 4,
    'bed': 5,
    'toilet': 6,
    'tv_monitor': 7,
    'table': 8,
    'refrigerator': 9,
    'sink': 10,
    'stairs': 11,
    'fireplace': 12
}

# mp_categories_mapping = [4, 11, 15, 12, 19, 23, 6, 7, 15, 38, 40, 28, 29, 8, 17]

mp_categories_mapping = [4, 11, 15, 12, 19, 23, 26, 24, 28, 38, 21, 16, 14, 6, 16]
mp_categories_mapping21 = [4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 26, 27, 28, 34, 35, 39]

hm3d_category = [
        "chair",
        "sofa",
        "plant",
        "bed",
        "toilet",
        "tv",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
]

hm3d_target_to_label = {
    "chair": "chair",
    "sofa": "sofa",
    "plant": "plant",
    "bed": "bed",
    "toilet": "toilet",
    "tv": "tv",
    "table": "table",
}

mp3d_target_to_label = {
    "chair": "chair",
    "sofa": "sofa",
    "plant": "plant",
    "bed": "bed",
    "toilet": "toilet",
    "tv": "tv",
    "table": "table",
}

hm3d_label_to_idx = {label.lower(): idx for idx, label in enumerate(hm3d_category)}

cleaned_mapping = {
  "Unknown": "IGNORE",
  "air conditioner": "appliances",
  "air conditioning": "appliances",
  "air duct": "IGNORE",
  "air freshener": "IGNORE",
  "air vent": "IGNORE",
  "alarm": "IGNORE",
  "alarm clock": "IGNORE",
  "alarm control": "IGNORE",
  "amplifier": "IGNORE",
  "antique clock": "IGNORE",
  "apple": "IGNORE",
  "appliance": "appliances",
  "arcade": "IGNORE",
  "arcade game": "IGNORE",
  "arch": "IGNORE",
  "armchair": "chair",
  "art frame": "IGNORE",
  "artwork": "IGNORE",
  "attic door": "IGNORE",
  "attic hatch": "IGNORE",
  "baby changing table": "table",
  "backpack": "IGNORE",
  "bag": "IGNORE",
  "bags": "IGNORE",
  "ball": "IGNORE",
  "balustrade": "IGNORE",
  "banister": "IGNORE",
  "bar": "IGNORE",
  "bar chair": "chair",
  "barrel": "IGNORE",
  "base": "IGNORE",
  "baseball bat": "IGNORE",
  "basin": "sink",
  "basket": "IGNORE",
  "basket of something": "IGNORE",
  "basket of towels": "towel",
  "basket with books": "IGNORE",
  "basketballs": "IGNORE",
  "baskets": "IGNORE",
  "bath": "IGNORE",
  "bath cabinet": "IGNORE",
  "bath faucet": "IGNORE",
  "bath mat": "IGNORE",
  "bath side table": "table",
  "bath sink": "sink",
  "bath towel": "towel",
  "bath towels": "towel",
  "bath tub": "bathtub",
  "bath utensil": "IGNORE",
  "bath wall": "IGNORE",
  "bathrobe": "IGNORE",
  "bathroom accessory": "IGNORE",
  "bathroom art": "IGNORE",
  "bathroom cabinet": "IGNORE",
  "bathroom counter": "IGNORE",
  "bathroom glass": "IGNORE",
  "bathroom mat": "IGNORE",
  "bathroom shelf": "IGNORE",
  "bathroom stuff": "IGNORE",
  "bathroom towel": "towel",
  "bathroom utensil": "IGNORE",
  "bathtub": "bathtub",
  "bathtub platform": "bathtub",
  "bathtub tap": "IGNORE",
  "bathtub utensil": "bathtub",
  "beam": "IGNORE",
  "bean bag chair": "chair",
  "bed": "bed",
  "bed light": "IGNORE",
  "bed sheet": "IGNORE",
  "bed small": "bed",
  "bed stand": "IGNORE",
  "bed table": "table",
  "bedframe": "bed",
  "bedroom ceiling": "IGNORE",
  "bedroom table": "table",
  "bedside cabinet": "IGNORE",
  "bedside lamp": "IGNORE",
  "bedside table": "table",
  "bell": "IGNORE",
  "bench": "chair",
  "bicycle": "IGNORE",
  "bicycle helmets": "IGNORE",
  "bin": "IGNORE",
  "binder": "IGNORE",
  "blanket": "IGNORE",
  "blankets": "IGNORE",
  "blinds": "IGNORE",
  "blouse": "IGNORE",
  "board": "IGNORE",
  "board games": "IGNORE",
  "boards": "IGNORE",
  "boiler": "IGNORE",
  "bonsai tree": "plant",
  "book": "IGNORE",
  "book rack": "IGNORE",
  "bookshelf": "IGNORE",
  "bookstand": "IGNORE",
  "bottle": "IGNORE",
  "bottle of detergent": "IGNORE",
  "bottle of soap": "IGNORE",
  "bottles": "IGNORE",
  "bottles of water": "IGNORE",
  "bottles of wine": "IGNORE",
  "bouquet": "plant",
  "bowl": "IGNORE",
  "bowl of fruit": "IGNORE",
  "bowl of sweets": "IGNORE",
  "box": "IGNORE",
  "box of food": "IGNORE",
  "box of tissue": "IGNORE",
  "box of tissues": "IGNORE",
  "box pen": "IGNORE",
  "box with books": "IGNORE",
  "box with photos": "IGNORE",
  "boxes": "IGNORE",
  "boxes with books": "IGNORE",
  "briefcase": "IGNORE",
  "brochure": "IGNORE",
  "broom": "IGNORE",
  "brush": "IGNORE",
  "bucket": "IGNORE",
  "bulletin board": "IGNORE",
  "bust": "IGNORE",
  "cabinet": "IGNORE",
  "cabinet clutter": "IGNORE",
  "cabinet door": "IGNORE",
  "cabinet table": "table",
  "cable": "IGNORE",
  "cables": "IGNORE",
  "calendar": "IGNORE",
  "camera": "IGNORE",
  "camping chair": "chair",
  "can": "IGNORE",
  "can of paint": "IGNORE",
  "candle": "IGNORE",
  "candle holder": "IGNORE",
  "candlestick": "IGNORE",
  "canister": "IGNORE",
  "cans": "IGNORE",
  "cans of paint": "IGNORE",
  "car": "IGNORE",
  "card": "IGNORE",
  "cardboard": "IGNORE",
  "cardboard box": "IGNORE",
  "carpet": "IGNORE",
  "carpet roll": "IGNORE",
  "cart": "IGNORE",
  "case": "IGNORE",
  "casket": "IGNORE",
  "ceiling": "IGNORE",
  "ceiling arch": "IGNORE",
  "ceiling boarder": "IGNORE",
  "ceiling decorative lamp": "IGNORE",
  "ceiling dome": "IGNORE",
  "ceiling door": "IGNORE",
  "ceiling duct": "IGNORE",
  "ceiling fan": "IGNORE",
  "ceiling fan lamp": "IGNORE",
  "ceiling fixture": "IGNORE",
  "ceiling ladder": "IGNORE",
  "ceiling lamp": "IGNORE",
  "ceiling light": "IGNORE",
  "ceiling light fixture connection": "IGNORE",
  "ceiling lower": "IGNORE",
  "ceiling molding": "IGNORE",
  "ceiling support": "IGNORE",
  "ceiling under staircase": "IGNORE",
  "ceiling under stairs": "stairs",
  "ceiling vent": "IGNORE",
  "ceiling wall": "IGNORE",
  "ceiling window": "IGNORE",
  "chair": "chair",
  "chair stand": "IGNORE",
  "chandelier": "IGNORE",
  "chassis": "IGNORE",
  "chess": "IGNORE",
  "chest": "IGNORE",
  "chest bench": "chair",
  "chest drawer": "IGNORE",
  "chest of drawers": "chest_of_drawers",
  "chimney": "IGNORE",
  "christmas tree": "plant",
  "circular sofa": "sofa",
  "cleaning clutter": "IGNORE",
  "clock": "IGNORE",
  "closet": "IGNORE",
  "closet area for hanging clothes": "IGNORE",
  "closet door": "IGNORE",
  "closet mirror wall": "IGNORE",
  "closet rod": "IGNORE",
  "closet shelf": "IGNORE",
  "closet storage area": "IGNORE",
  "cloth": "IGNORE",
  "cloth hanger": "IGNORE",
  "cloth hangers": "IGNORE",
  "clothes": "IGNORE",
  "clothes container": "IGNORE",
  "clothes dryer": "IGNORE",
  "clothes hanger": "IGNORE",
  "clothes hanger rod": "IGNORE",
  "clothes on shelf": "IGNORE",
  "clothes rack": "IGNORE",
  "clutter": "IGNORE",
  "coaster": "IGNORE",
  "coat hanger": "IGNORE",
  "coffee machine": "appliances",
  "coffee maker": "appliances",
  "coffee mug": "IGNORE",
  "coffee table": "table",
  "column": "IGNORE",
  "compound wall": "IGNORE",
  "compressor": "IGNORE",
  "computer": "IGNORE",
  "computer chair": "chair",
  "computer desk": "IGNORE",
  "computer equipment": "IGNORE",
  "computer mouse": "IGNORE",
  "computer tower": "IGNORE",
  "condiment": "IGNORE",
  "container": "IGNORE",
  "containers": "IGNORE",
  "control panel": "IGNORE",
  "controller": "IGNORE",
  "cook book": "IGNORE",
  "cooker": "IGNORE",
  "cosmetic": "IGNORE",
  "cosmetics": "IGNORE",
  "couch": "sofa",
  "counter": "IGNORE",
  "countertop": "IGNORE",
  "cover": "IGNORE",
  "crate": "IGNORE",
  "crib": "bed",
  "cross": "IGNORE",
  "cuddly toy": "IGNORE",
  "cup": "IGNORE",
  "curtain": "IGNORE",
  "curtain bar": "IGNORE",
  "curtain rail": "IGNORE",
  "curtain rod": "IGNORE",
  "curtain rod cover": "IGNORE",
  "cushion": "IGNORE",
  "cutting board": "IGNORE",
  "decor": "IGNORE",
  "decoration": "IGNORE",
  "decorative bowl": "IGNORE",
  "decorative frame": "IGNORE",
  "decorative mask": "IGNORE",
  "decorative plant": "plant",
  "decorative plate": "IGNORE",
  "decorative window": "IGNORE",
  "desk": "IGNORE",
  "desk chair": "chair",
  "desk clutter": "IGNORE",
  "desk door": "IGNORE",
  "desk lamp": "IGNORE",
  "detergent bottle": "IGNORE",
  "device": "IGNORE",
  "dining chair": "chair",
  "dining table": "table",
  "diploma": "IGNORE",
  "dish dryer": "IGNORE",
  "dishrag": "IGNORE",
  "dishwasher": "appliances",
  "display": "IGNORE",
  "display cabinet": "IGNORE",
  "display table": "table",
  "document": "IGNORE",
  "dog bed": "bed",
  "doily": "IGNORE",
  "door": "IGNORE",
  "door frame": "IGNORE",
  "door handle": "IGNORE",
  "door hinge": "IGNORE",
  "door knob": "IGNORE",
  "door stopper": "IGNORE",
  "door window": "IGNORE",
  "door/window": "IGNORE",
  "door/window frame": "IGNORE",
  "doormat": "IGNORE",
  "doorpost": "IGNORE",
  "doorstep": "IGNORE",
  "drawer": "IGNORE",
  "drawer cabinet": "chest_of_drawers",
  "drawers": "IGNORE",
  "drawing": "IGNORE",
  "dresser": "chest_of_drawers",
  "dressing table": "table",
  "dried flowers": "plant",
  "drum": "IGNORE",
  "duct": "IGNORE",
  "dumbbell": "IGNORE",
  "dustbin": "IGNORE",
  "dvd player": "IGNORE",
  "easel": "IGNORE",
  "electric box": "IGNORE",
  "electric cable": "IGNORE",
  "electric device": "IGNORE",
  "electric freshener": "IGNORE",
  "electric toothbrush": "IGNORE",
  "electric wire": "IGNORE",
  "electrical box": "IGNORE",
  "electrical controller": "IGNORE",
  "electrical device": "IGNORE",
  "electrical switchboard": "IGNORE",
  "elevator door": "IGNORE",
  "end table": "table",
  "entertainment set": "IGNORE",
  "entrance arch": "IGNORE",
  "exercise ball": "IGNORE",
  "exercise bike": "IGNORE",
  "exercise equipment": "IGNORE",
  "exercise machine": "IGNORE",
  "exercise mat": "IGNORE",
  "exhibition panel": "IGNORE",
  "exhibition picture": "IGNORE",
  "exhibition table": "table",
  "exhibition window": "IGNORE",
  "exhibition window frame": "IGNORE",
  "exit sign": "IGNORE",
  "fan": "IGNORE",
  "faucet": "IGNORE",
  "figure": "IGNORE",
  "figurine": "IGNORE",
  "file binder": "IGNORE",
  "file cabinet": "IGNORE",
  "fire alarm": "IGNORE",
  "fire detector": "IGNORE",
  "fire extinguisher": "IGNORE",
  "fire screen": "IGNORE",
  "fire sprinkler": "IGNORE",
  "fireplace": "fireplace",
  "fireplace shelf": "fireplace",
  "fireplace tool set": "fireplace",
  "fireplace utensil": "fireplace",
  "fireplace wall": "fireplace",
  "firewood": "IGNORE",
  "firewood holder": "IGNORE",
  "fishing pole": "IGNORE",
  "flag": "IGNORE",
  "floor": "IGNORE",
  "floor lamp": "IGNORE",
  "floor mat": "IGNORE",
  "floor stand": "IGNORE",
  "floor vent": "IGNORE",
  "flower": "plant",
  "flower stand": "IGNORE",
  "flower vase": "IGNORE",
  "flowerpot": "IGNORE",
  "folded chair": "chair",
  "folder": "IGNORE",
  "folding chair": "chair",
  "food": "IGNORE",
  "food stand": "IGNORE",
  "football": "IGNORE",
  "footrest": "IGNORE",
  "footstool": "chair",
  "frame": "IGNORE",
  "framed text": "IGNORE",
  "freezer": "appliances",
  "fridge": "appliances",
  "fruit": "IGNORE",
  "fruit bowl": "IGNORE",
  "fume cupboard": "appliances",
  "fur": "IGNORE",
  "fur carpet": "IGNORE",
  "furnace": "appliances",
  "furniture": "IGNORE",
  "fuse box": "IGNORE",
  "garage door": "IGNORE",
  "garage door opener motor": "IGNORE",
  "garage door opener railing": "IGNORE",
  "garage door railing": "IGNORE",
  "glass": "IGNORE",
  "glasses": "IGNORE",
  "globe": "IGNORE",
  "globe stand": "IGNORE",
  "gloves": "IGNORE",
  "glue": "IGNORE",
  "grate": "IGNORE",
  "gravel": "IGNORE",
  "grill": "IGNORE",
  "guitar": "IGNORE",
  "gun": "IGNORE",
  "hammer": "IGNORE",
  "hammock": "IGNORE",
  "hand soap": "IGNORE",
  "handbag": "IGNORE",
  "handkerchiefs": "IGNORE",
  "handle": "IGNORE",
  "handrail": "IGNORE",
  "hanger": "IGNORE",
  "hanging clothes": "IGNORE",
  "hat": "IGNORE",
  "headboard": "bed",
  "hearth": "fireplace",
  "heater": "appliances",
  "heater piping": "appliances",
  "high shelf": "IGNORE",
  "highchair": "chair",
  "holy cross": "IGNORE",
  "hook": "IGNORE",
  "hose": "IGNORE",
  "hot water/cold water knob": "IGNORE",
  "hunting trophy": "IGNORE",
  "ice maker": "appliances",
  "information": "IGNORE",
  "iron": "IGNORE",
  "iron board": "IGNORE",
  "ironing board": "IGNORE",
  "island": "IGNORE",
  "jacket": "IGNORE",
  "jar": "IGNORE",
  "jewelry box": "IGNORE",
  "jug": "IGNORE",
  "kettle": "appliances",
  "keyboard": "IGNORE",
  "keys": "IGNORE",
  "kitchen appliance": "appliances",
  "kitchen cabinet": "IGNORE",
  "kitchen cabinet door": "IGNORE",
  "kitchen cabinet drawer": "IGNORE",
  "kitchen cabinet lower": "IGNORE",
  "kitchen counter": "IGNORE",
  "kitchen countertop item": "IGNORE",
  "kitchen countertop items": "IGNORE",
  "kitchen decoration": "IGNORE",
  "kitchen extractor": "appliances",
  "kitchen island": "IGNORE",
  "kitchen lower cabinet": "IGNORE",
  "kitchen shelf": "IGNORE",
  "kitchen sink cabinet": "IGNORE",
  "kitchen top": "IGNORE",
  "kitchen utensil": "IGNORE",
  "kitchen utensils": "IGNORE",
  "kitchen wall": "IGNORE",
  "knife": "IGNORE",
  "knife holder": "IGNORE",
  "knob": "IGNORE",
  "l-shaped sofa": "sofa",
  "lace doily": "IGNORE",
  "ladder": "IGNORE",
  "lamp": "IGNORE",
  "lamp table": "table",
  "lampshade": "IGNORE",
  "laptop": "IGNORE",
  "laundry basket": "IGNORE",
  "laundry machine": "appliances",
  "led tv": "tv",
  "ledge": "IGNORE",
  "level": "IGNORE",
  "lid": "IGNORE",
  "light": "IGNORE",
  "light fixture": "IGNORE",
  "light switch": "IGNORE",
  "lighter": "IGNORE",
  "lighting fixture": "IGNORE",
  "liquid container": "IGNORE",
  "liquid soap": "IGNORE",
  "lounge chair": "chair",
  "lounger": "chair",
  "luggage": "IGNORE",
  "machine": "IGNORE",
  "magazine": "IGNORE",
  "magazines": "IGNORE",
  "mantle": "fireplace",
  "map": "IGNORE",
  "mascot": "IGNORE",
  "mat": "IGNORE",
  "measuring tape": "IGNORE",
  "media console": "table",
  "microwave": "appliances",
  "mini fridge": "appliances",
  "mirror": "IGNORE",
  "mirror frame": "IGNORE",
  "model": "IGNORE",
  "molding": "IGNORE",
  "monitor": "IGNORE",
  "motion detector": "IGNORE",
  "mug": "IGNORE",
  "music player": "IGNORE",
  "newspaper": "IGNORE",
  "niche": "IGNORE",
  "nightstand": "table",
  "note": "IGNORE",
  "notebook": "IGNORE",
  "notebooks": "IGNORE",
  "object": "IGNORE",
  "objects": "IGNORE",
  "office chair": "chair",
  "ornament": "IGNORE",
  "ottoman": "chair",
  "outlet": "IGNORE",
  "oven": "appliances",
  "oven and stove": "appliances",
  "oven vent": "appliances",
  "pad": "IGNORE",
  "painting": "IGNORE",
  "painting frame": "IGNORE",
  "painting roll": "IGNORE",
  "painting rolls": "IGNORE",
  "painting tray": "IGNORE",
  "pan": "IGNORE",
  "panel": "IGNORE",
  "paneling": "IGNORE",
  "paper": "IGNORE",
  "paper towel": "IGNORE",
  "paper towel dispenser": "IGNORE",
  "parapet": "IGNORE",
  "partition": "IGNORE",
  "patio chair": "chair",
  "pc tower": "IGNORE",
  "pedestal": "IGNORE",
  "pen": "IGNORE",
  "pen cup": "IGNORE",
  "pendant": "IGNORE",
  "perfume": "IGNORE",
  "phone": "IGNORE",
  "photo": "IGNORE",
  "photo mount": "IGNORE",
  "piano": "IGNORE",
  "piano bench": "chair",
  "piano stool": "chair",
  "picture": "IGNORE",
  "picture frame": "IGNORE",
  "pile of magazines": "IGNORE",
  "pillar": "IGNORE",
  "pillow": "IGNORE",
  "pipe": "IGNORE",
  "pitcher": "IGNORE",
  "plank": "IGNORE",
  "plant": "plant",
  "plate": "IGNORE",
  "plates": "IGNORE",
  "platform": "IGNORE",
  "pliers": "IGNORE",
  "plumbing": "IGNORE",
  "plunger": "IGNORE",
  "plush toy": "IGNORE",
  "podium": "IGNORE",
  "poles": "IGNORE",
  "pool stick": "IGNORE",
  "poster": "IGNORE",
  "pot": "IGNORE",
  "pot lid": "IGNORE",
  "potty": "toilet",
  "pouches": "IGNORE",
  "pouffe": "chair",
  "printer": "IGNORE",
  "projector": "IGNORE",
  "projector screen": "IGNORE",
  "rack": "IGNORE",
  "rack with shoes": "IGNORE",
  "radiator": "IGNORE",
  "radio": "IGNORE",
  "rafter": "IGNORE",
  "ragdoll": "IGNORE",
  "ragdoll cat": "IGNORE",
  "rail": "IGNORE",
  "railing": "IGNORE",
  "range hood": "appliances",
  "recessed wall": "IGNORE",
  "recliner": "chair",
  "record player": "IGNORE",
  "recuperator": "IGNORE",
  "recycle bin": "IGNORE",
  "refrigerator": "appliances",
  "refrigerator cabinet": "appliances",
  "remote control": "IGNORE",
  "robe": "IGNORE",
  "rock": "IGNORE",
  "rocking chair": "chair",
  "rod": "IGNORE",
  "rods": "IGNORE",
  "roll": "IGNORE",
  "roomba": "IGNORE",
  "rug": "IGNORE",
  "ruler": "IGNORE",
  "safe": "IGNORE",
  "salt and pepper": "IGNORE",
  "sauna bowl": "IGNORE",
  "sauna heater": "appliances",
  "saw": "IGNORE",
  "scale": "IGNORE",
  "sconce": "IGNORE",
  "scoop": "IGNORE",
  "screen": "IGNORE",
  "screwdriver": "IGNORE",
  "sculpture": "IGNORE",
  "seat": "chair",
  "security camera": "IGNORE",
  "self-closing mechanism": "IGNORE",
  "sensor": "IGNORE",
  "sewing box": "IGNORE",
  "sewing machine": "IGNORE",
  "shade": "IGNORE",
  "shades": "IGNORE",
  "shampoo": "IGNORE",
  "sheet": "IGNORE",
  "shelf": "IGNORE",
  "shelf / cabinet": "IGNORE",
  "shelf cubby": "IGNORE",
  "shelf with clutter": "IGNORE",
  "shelving": "IGNORE",
  "ship model": "IGNORE",
  "shirt": "IGNORE",
  "shoe": "IGNORE",
  "shoes": "IGNORE",
  "shoes on shelf": "IGNORE",
  "shoulder bag": "IGNORE",
  "shovel": "IGNORE",
  "shower": "shower",
  "shower bar": "IGNORE",
  "shower bench": "chair",
  "shower cabin": "shower",
  "shower cabinet": "shower",
  "shower caddy": "IGNORE",
  "shower ceiling": "shower",
  "shower ceiling lamp": "shower",
  "shower cosmetics": "shower",
  "shower curtain": "IGNORE",
  "shower curtain bar": "IGNORE",
  "shower curtain rod": "IGNORE",
  "shower dial": "shower",
  "shower door": "IGNORE",
  "shower door frame": "IGNORE",
  "shower floor": "shower",
  "shower frame": "shower",
  "shower glass": "shower",
  "shower handle": "shower",
  "shower hose": "IGNORE",
  "shower hose/head": "IGNORE",
  "shower knob": "IGNORE",
  "shower mat": "IGNORE",
  "shower pipe": "shower",
  "shower rail": "IGNORE",
  "shower rod": "IGNORE",
  "shower seat": "shower",
  "shower shelf": "shower",
  "shower soap shelf": "IGNORE",
  "shower stall": "shower",
  "shower step": "shower",
  "shower tap": "IGNORE",
  "shower tub": "bathtub",
  "shower valve": "IGNORE",
  "shower wall": "shower",
  "shower wall cubby": "shower",
  "shower-bath cabinet": "shower",
  "showerhead": "IGNORE",
  "shredder": "IGNORE",
  "shutter": "IGNORE",
  "side table": "table",
  "sideboard": "table",
  "sign": "IGNORE",
  "silicone gun": "IGNORE",
  "sink": "sink",
  "sink cabinet": "IGNORE",
  "sink pipe": "IGNORE",
  "sink tap": "IGNORE",
  "sitting bench": "chair",
  "skateboard": "IGNORE",
  "slab": "IGNORE",
  "sleeping bag": "IGNORE",
  "sliding door": "IGNORE",
  "sliding glass door": "IGNORE",
  "slippers": "IGNORE",
  "small table/stand": "table",
  "smoke alarm": "IGNORE",
  "smoke detector": "IGNORE",
  "snack": "IGNORE",
  "soap": "IGNORE",
  "soap bottle": "IGNORE",
  "soap dish": "IGNORE",
  "soap dispenser": "IGNORE",
  "socket": "IGNORE",
  "sofa": "sofa",
  "sofa chair": "sofa",
  "sofa seat": "sofa",
  "sofa set": "sofa",
  "spa bench": "chair",
  "spatula": "IGNORE",
  "speaker": "IGNORE",
  "spice rack": "IGNORE",
  "spray": "IGNORE",
  "sprinkler": "IGNORE",
  "square": "IGNORE",
  "stack of jackets": "IGNORE",
  "stack of papers": "IGNORE",
  "stained glass": "IGNORE",
  "stair": "stairs",
  "stair handle": "IGNORE",
  "stair step": "stairs",
  "stair wall": "IGNORE",
  "staircase handrail": "IGNORE",
  "staircase trim": "IGNORE",
  "staircase wall": "IGNORE",
  "stairs": "stairs",
  "stairs railing": "IGNORE",
  "stand": "IGNORE",
  "star": "IGNORE",
  "statue": "IGNORE",
  "statue/art": "IGNORE",
  "step": "IGNORE",
  "step stool": "chair",
  "stereo": "IGNORE",
  "stick": "IGNORE",
  "stonework": "IGNORE",
  "stool": "chair",
  "storage box": "IGNORE",
  "storage cabinet": "IGNORE",
  "storage shelving": "IGNORE",
  "stove": "appliances",
  "stovetop": "appliances",
  "stuffed animal": "IGNORE",
  "support": "IGNORE",
  "support beam": "IGNORE",
  "surface": "IGNORE",
  "surfboard": "IGNORE",
  "switch": "IGNORE",
  "table": "table",
  "table cloth": "IGNORE",
  "table lamp": "IGNORE",
  "table stand": "table",
  "table tennis table": "table",
  "tablet": "IGNORE",
  "tank": "IGNORE",
  "tap": "IGNORE",
  "teapot": "IGNORE",
  "telephone": "IGNORE",
  "terrace": "IGNORE",
  "thermostat": "IGNORE",
  "throw blanket": "IGNORE",
  "tissue box": "IGNORE",
  "toaster": "appliances",
  "toilet": "toilet",
  "toilet brush": "IGNORE",
  "toilet brush holder": "IGNORE",
  "toilet cleaner": "IGNORE",
  "toilet paper": "IGNORE",
  "toilet paper dispenser": "IGNORE",
  "toilet seat": "IGNORE",
  "toiletry": "IGNORE",
  "tool": "IGNORE",
  "tool box": "IGNORE",
  "tool rack": "IGNORE",
  "towel": "towel",
  "towel bar": "IGNORE",
  "towel ring": "IGNORE",
  "toy": "IGNORE",
  "training mat": "IGNORE",
  "trash can": "IGNORE",
  "trashcan": "IGNORE",
  "tray": "IGNORE",
  "treadmill": "IGNORE",
  "trinket": "IGNORE",
  "tv": "tv",
  "tv remote": "IGNORE",
  "tv stand": "table",
  "umbrella": "IGNORE",
  "unknown": "IGNORE",
  "unknown/remove": "IGNORE",
  "urinal": "toilet",
  "utensil": "IGNORE",
  "vacuum cleaner": "appliances",
  "vase": "IGNORE",
  "vent": "IGNORE",
  "ventilation": "IGNORE",
  "ventilation hood": "appliances",
  "vessel": "IGNORE",
  "vinyl records": "IGNORE",
  "wall": "IGNORE",
  "wall cabinet": "IGNORE",
  "wall clock": "IGNORE",
  "wall control": "IGNORE",
  "wall cubby": "IGNORE",
  "wall electronics": "IGNORE",
  "wall hanger": "IGNORE",
  "wall hanging decoration": "IGNORE",
  "wall lamp": "IGNORE",
  "wall panel": "IGNORE",
  "wall post": "IGNORE",
  "wall shelf": "IGNORE",
  "wall sign": "IGNORE",
  "wall soap shelf": "IGNORE",
  "wall toilet paper": "IGNORE",
  "wall tv": "tv",
  "wardrobe": "IGNORE",
  "wash cabinet": "IGNORE",
  "washbasin": "sink",
  "washbasin counter": "sink",
  "washcloth": "IGNORE",
  "washer-dryer": "appliances",
  "washing machine": "appliances",
  "washing machine and dryer": "appliances",
  "washing powder": "IGNORE",
  "washing stuff": "IGNORE",
  "watch": "IGNORE",
  "water heater": "appliances",
  "watering can": "IGNORE",
  "weight": "IGNORE",
  "weight bench": "chair",
  "wheel": "IGNORE",
  "whine shelf": "IGNORE",
  "wifi router": "IGNORE",
  "window": "IGNORE",
  "window curtain": "IGNORE",
  "window frame": "IGNORE",
  "window glass": "IGNORE",
  "window seat": "chair",
  "window shade": "IGNORE",
  "window shutter": "IGNORE",
  "window shutters": "IGNORE",
  "window valence": "IGNORE",
  "window/door": "IGNORE",
  "wine": "IGNORE",
  "wine bottle": "IGNORE",
  "wine rack": "IGNORE",
  "wood": "IGNORE",
  "worktop": "IGNORE",
  "wreath": "plant",
  "wrench": "IGNORE"
}

mp3d_category = [
    'chair', #g
    'table', #g
    'picture', #b
    'cabinet', # in resnet
    'cushion', # in resnet
    'couch', #g
    'bed', #g
    'drawer', #b in resnet
    'plant', #g
    'sink', #g
    'toilet', #g
    'stool', #b
    'towel', #b in resnet
    'tv', #g
    'shower', #b
    'bathtub', #b in resnet
    'counter', #b isn't this table?
    'fireplace',
    'gym equipment',
    'seating',
    'clothes', # in resnet
    'background'
]

sample_categories = { "chair": 0, "sofa": 1, "plant": 2, "bed": 3, "toilet": 4, "tv": 5, "table": 6 }

mp3d_habitat_labels = {
            'chair': 0, #g
            'table': 1, #g
            'picture':2, #b
            'cabinet':3, # in resnet
            'cushion':4, # in resnet
            'sofa':5, #g
            'bed':6, #g
            'chest_of_drawers':7, #b in resnet
            'plant':8, #g
            'sink':9, #g
            'toilet':10, #g
            'stool':11, #b
            'towel':12, #b in resnet
            'tv_monitor':13, #g
            'shower':14, #b
            'bathtub':15, #b in resnet
            'counter':16, #b isn't this table?
            'fireplace':17,
            'gym_equipment':18,
            'seating':19,
            'clothes':20, # in resnet
            'background': 21
}

gibson_coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

object_category = [
    'chair', #g
    'table', #g
    'picture', #b
    'cabinet', # in resnet
    'cushion', # in resnet
    'sofa', #g
    'bed', #g
    'chest_of_drawers', #b in resnet
    'plant', #g
    'sink', #g
    'toilet', #g
    'stool', #b
    'towel', #b in resnet
    'tv_monitor', #g
    'shower', #b
    'bathtub', #b in resnet
    'counter', #b isn't this table?
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes', # in resnet
    'background'
]

mp3d_label_to_idx = {label.lower(): idx for idx, label in enumerate(object_category)}

object_category_copy = [
    # --- goal categories (DO NOT CHANGE)
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv",
    "couch",
    # --- add as many categories in between as you wish, which are used for LLM reasoning
    # "table",
    "desk",
    "refrigerator",
    "sink",
    "bathtub",
    "shower",
    "towel",
    "painting",
    "trashcan",
    # --- stairs must be put here (DO NOT CHANGE)
    "stairs",
    # --- void category (DO NOT CHANGE)
    "void"
]

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
    # NEW COLORS
    0.66, 0.9400000000000001, 0.9531999999999998,
    0.7600000000000001, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.9531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
    0.66, 0.9400000000000001, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.9581999999999998,
    0.8882000000000001, 0.9400000000000001, 0.9531999999999998,
    0.7832000000000001, 0.9400000000000001, 0.9581999999999998,
    0.6782000000000001, 0.9400000000000001, 0.9531999999999998,
    0.66, 0.9400000000000001, 0.7618000000000001,
    0.66, 0.9400000000000001, 0.9661999999999998,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9661999999999998,
    0.9400000000000001, 0.66, 0.7668000000000001,
    0.66, 0.9661999999999998, 0.9400000000000001,
    0.7832000000000001, 0.9661999999999998, 0.66,
    0.9400000000000001, 0.8531999999999998, 0.66,
    0.66, 0.9661999999999998, 0.9681999999999998,
    0.8882000000000001, 0.66, 0.9661999999999998,
    0.66, 0.7657999999999998, 0.9661999999999998,
]
