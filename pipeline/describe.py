# -*- coding: utf-8 -*-
"""What an annotation on the clip page's Objects step means, from its free-text description ("what") and type.

  parse("white pillar by the door", "other") -> {"mode": "model", "kind": "pillar", "label": "pillar", "role": None}
  parse("plastic bag held in his left hand", "other") -> {"mode": "held", "kind": "bag", "label": "bag", "side": "L", ...}
  parse("old wooden sculpture of a horse", "other") -> {"mode": "shape", "kind": None, "label": "sculpture", ...}

mode "model": built as that props.py library model (doors, pillars, tables, cars... - things with a regular shape);
mode "shape": built from its measured surface inside the box (relief.py: plants, curtains, statues, and anything the
              description does not name);
mode "held":  carried by a person (in a hand, or worn on the back): placed on the fitted hand / back by
              held_objects.py, never as a fixed set piece.
A type picked in the list (anything but "other" / "box") wins over the description, unless the description says it
is held.  Colours follow the colour card (colour_card.py) by meaning; words like "interactive", "distant door",
"special", "foreground" ask for that meaning ("role")."""
import re

HELD = re.compile(r"\b(held|holds?|holding|carr(?:y|ies|ied|ying)|in (?:his|her|their|the|a|one)?\s*(?:left |right )?hands?|"
                  r"on (?:his|her|their) (?:back|shoulder)|wears?|wearing|worn|grip(?:s|ped|ping)?|clutch(?:es|ed|ing)?)\b", re.I)
WORN = re.compile(r"\b(backpack|rucksack|on (?:his|her|their) back|wears?|wearing|worn)\b", re.I)
LEFT = re.compile(r"\bleft\b", re.I)
RIGHT = re.compile(r"\bright\b", re.I)
# irregular things: built from their measured surface
SOFT = ["potted plant", "house plant", "plant", "flowers", "flower", "bouquet", "curtains", "curtain", "drapes", "drape",
        "statue", "sculpture", "figure", "bust", "vase", "cushion", "pillow", "blanket", "towel", "clothes", "clothing",
        "coat", "jacket", "dress", "bag", "sack", "heap", "pile", "debris", "rubble", "animal", "dog", "cat", "horse",
        "bicycle", "bike", "motorbike", "motorcycle", "boat", "chandelier", "fountain", "basket", "toy", "plush",
        "teddy", "food", "fruit", "flag", "banner", "bush", "shrub", "hedge", "leaves", "foliage", "palm"]


def _types():
    import props
    words = {}
    for t in props.TYPES:
        if t != "other":
            words[t.replace("_", " ")] = t
    for a, t in props.ALIASES.items():
        if t in props.TYPES and t != "other":
            words[a.replace("_", " ")] = t
    return words


def _find(text, phrases):
    """the phrase of <phrases> that names the thing in <text>: whole words (plurals too), the one ending last (English
    puts the thing last: a desk lamp is a lamp, a coffee table a table), the longer one on a tie -> (phrase, start)"""
    best = None
    for p in phrases:
        for m in re.finditer(r"\b%s(?:e?s)?\b" % re.escape(p), text):
            key = (m.end(), len(p))
            if best is None or key > best[2]:
                best = (p, m.start(), key)
    return best[:2] if best else None


def parse(what, typed=None):
    text = re.sub(r"\s+", " ", str(what or "").lower()).strip()
    typed = (str(typed or "").strip().lower() or "other")
    import colour_card
    out = {"label": None, "kind": None, "mode": "shape", "role": colour_card.word_role(text), "side": None}
    types = _types()
    head = re.split(r"\b(?:on|in|inside|with|by|of|next to|near|under|behind|beside|at|against|above|below|over|from|"
                    r"held|holding|carried|carrying|that|which)\b", text)[0].strip()   # "the thing on the shelf": a thing
    where = head if head else text
    t_hit = _find(where, types)
    s_hit = _find(where, SOFT)
    if not head and not (t_hit or s_hit):
        t_hit, s_hit = _find(text, types), _find(text, SOFT)
    noun = None                                                   # the thing: the word ending last ("plant table" is a table)
    if t_hit and s_hit:
        te, se = (t_hit[1] + len(t_hit[0]), len(t_hit[0])), (s_hit[1] + len(s_hit[0]), len(s_hit[0]))
        noun = ("type", t_hit[0]) if te > se else ("soft", s_hit[0])
    elif t_hit:
        noun = ("type", t_hit[0])
    elif s_hit:
        noun = ("soft", s_hit[0])
    if HELD.search(text) or (s_hit and s_hit[0] == "backpack") or "backpack" in text:
        out["mode"] = "held"
        out["worn"] = bool(WORN.search(text))
        out["side"] = "L" if LEFT.search(text) and not RIGHT.search(text) else ("R" if RIGHT.search(text) and not LEFT.search(text) else None)
        last = re.sub(r"[^a-z ]", "", head).split()[-1:] if head else []
        out["label"] = noun[1] if noun else (last[0] if last else ("backpack" if out["worn"] else "object"))
        out["kind"] = types.get(noun[1]) if noun and noun[0] == "type" else None
        return out
    if typed not in ("other", "box", "object", ""):              # a type picked in the list
        import props
        k = props.norm_type(typed)
        if k != "other":
            out.update(mode="shape" if k in ("plant_pot", "bush") else "model", kind=k, label=(noun[1] if noun else k.replace("_", " ")))
            return out
    if noun and noun[0] == "type":
        k = types[noun[1]]
        if k in ("plant_pot", "bush"):                            # (library plants look nothing like the real one)
            out.update(mode="shape", label=noun[1])
        else:
            out.update(mode="model", kind=k, label=noun[1])
    elif noun:
        out.update(mode="shape", label=noun[1])
    else:
        out.update(mode="shape", label=text[:30] or None)
    return out
