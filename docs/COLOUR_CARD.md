# Colour card (sRGB)

Every white-model video follows this card. The values live in `pipeline/colour_card.py` (`HEX`), and every render
applies them (`colour_card.apply`, run when the set is built and again just before Blender renders), so a clip
prepared before the card was added is recoloured when it is finished.

| Meaning | Colour | Hex | Used for |
|---|---|---|---|
| Main character | red | `#D63E3D` | first character's body and clothes |
| 2nd character | blue | `#2D6BD8` | second character |
| 3rd character | green | `#2E9D5A` | third character |
| 4th character | yellow | `#D9A014` | fourth character. For 4 or more, assign a stable colour per character and keep it for the whole clip. |
| Foreground object | brown | `#9A6B3B` | things in front of people, such as a table |
| Background | grey | `#B8B8B8` | background and surroundings, such as walls |
| **Nearby interactive object** | **orange** | `#DF872F` | something a person can interact with close by, such as a cup |
| **Distant interactive object** | **pink** | `#D4689E` | something interactive far from people, such as a distant door |
| Light source | white | `#F5F5F5` | marks a light and its position. Don't confuse it with the white head and chest marks. |
| Special object | purple | `#7A4EC2` | anything the rules don't cover |

## How the app decides

- **Characters**: the roles from the review (main, second, third, fourth) get red, blue, green, yellow. A 5th, 6th...
  person gets cyan `#20A8B8`, magenta `#B02878`, teal `#1F7A6E`, olive `#7E8B2A` (never a set-piece colour), the
  same one for the whole clip. The head and chest marks stay pure white.
- **Light source (white)**: lamps, chandeliers, sconces, street and ceiling lights.
- **Nearby interactive (orange)**: everything a person carries or holds (the held-object step), anything a hand comes
  within 12 cm of, and interactive things (door, cup, phone, bottle, chair, car, fridge, tv...) within 1 m of a person.
- **Distant interactive (pink)**: the same interactive things when no person comes within 1 m (a door across the room).
- **Foreground (brown)**: a set piece between the camera and a person: nearer to the camera than they are and
  overlapping them in the picture (a desk in front of someone sitting behind it).
- **Background (grey)**: walls, floor, ground, ceiling, far things, and furniture behind or beside the people.
- **Special (purple)**: only when asked for.

## Choosing by hand

On the clip page's Objects step, the description can ask for a meaning: "interactive", "distant door", "special",
"foreground", or name a light ("ceiling lamp"). The *interactive* tick makes it orange. In a hand-built set
(`env_claude.json`) an object's `"role"` (`bg`, `fg`, `near`, `far_i`, `lamp`, `special`) always wins.
