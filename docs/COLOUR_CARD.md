# Colour card (sRGB)

Every white-model video follows this card. The values live in `pipeline/colour_card.py` (`HEX`, `EXTRA`), and
`colour_card.apply` runs on the scene just before every Blender render. If it fails, the clip fails: a video is never
rendered in other colours. Colours are never sampled from the source video.

| Meaning | Colour | Hex | Used for |
|---|---|---|---|
| Main character | red | `#D63E3D` | first character's body |
| 2nd character | blue | `#2D6BD8` | second character |
| 3rd character | green | `#2E9D5A` | third character |
| 4th character | yellow | `#D9A014` | fourth character |
| 5th character on | cyan, magenta, teal, olive | `#20A8B8`, `#B02878`, `#1F7A6E`, `#7E8B2A` | one stable colour per character for the whole clip |
| Eye and chest bands | white | `#FFFFFF` | the marks on every character's head and chest |
| Background | grey | `#B8B8B8` | the floor (Stage 1); background and surroundings (Stage 2) |
| Foreground object | brown | `#9A6B3B` | Stage 2: things in front of people, such as a table |
| Nearby interactive object | orange | `#DF872F` | Stage 2: something a person can interact with close by, such as a cup |
| Distant interactive object | pink | `#D4689E` | Stage 2: something interactive far from people, such as a distant door |
| Light source | white | `#F5F5F5` | Stage 2: marks a light and its position |
| Special object | purple | `#7A4EC2` | Stage 2: anything the rules don't cover |

## Stage 1 (Movement)

- **Characters**: the fit numbers people by how large and how long they are on screen; the first four get red, blue,
  green and yellow. A 5th, 6th... person gets cyan, magenta, teal, olive (never a set-piece colour). A character
  with any other colour (an older fit or a `colours` list in `config.json`) is moved to the next free extra colour.
- **Marks**: the eye and chest bands are pure white.
- **Floor**: the only mesh of Stage 1, one ground disc per shot, is background: grey `#B8B8B8`.

## Stage 2 (Background)

Each placed object (`kind` / `label` = its model's type, e.g. `chair`) gets its role from these rules; a `"role"` in
`assets\objects\objects.json` always wins.

`colour_card.role_of` keeps the rules for set pieces: lamps and lights are light sources; interactive things
(door, cup, phone, chair, car, fridge, tv...) within 1 m of a person, or anything a hand comes within 12 cm of, are
nearby interactive, the same things farther away distant interactive; a piece between the camera and a person is
foreground; walls, floor, ceiling, far things and furniture behind people are background. A mesh's `"role"`
(`bg`, `fg`, `near`, `far_i`, `lamp`, `special`) always wins.
