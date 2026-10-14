"""Leaf: feasibility / affordance (可行性 / 可供性) from simulator object metadata.

Can this action be applied to this kind of object? Gold is what the simulator itself allows:
- AI2-THOR: the actionable properties of every object instance in the 195 scenes of
  ProcTHOR's AI2-THOR metadata dump (pickupable, openable, toggleable, sliceable, breakable,
  dirtyable, fillable, receptacle), and the simulator's hard-coded placement restrictions
  (which receptacle types each object type may be put in or on, SimObjType.cs);
- VirtualHome: the object properties the evolving-graph simulator reads
  (resources/properties_data.json), combined with the property check each action's executor
  performs (simulation/evolving_graph/execution.py), e.g. Grab needs GRABBABLE and Cut needs
  EATABLE and CUTTABLE.
Every question is about the kind of object alone, not about where the agent is or what it holds.
"""

from __future__ import annotations

import collections
import functools
import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "embodied/feasibility"


def spaced(name: str) -> str:
    """'TVStand' -> 'TV stand', 'CellPhone' -> 'cell phone', 'alarm_clock' -> 'alarm clock'."""
    if "_" in name or name.islower():
        return name.replace("_", " ")
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)
    return " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in words)


# ---------- AI2-THOR ----------

PROCTHOR = "https://raw.githubusercontent.com/allenai/procthor/53d5bd4c8c96a699e6a615dc390abb670cc9d353/procthor/databases/ai2thor-object-metadata.json"
SIMOBJTYPE = "https://raw.githubusercontent.com/allenai/ai2thor/24f79883b4889e3f0e6f4ae301808b9025872dfc/unity/Assets/Scripts/SimObjType.cs"
THOR_ACTIONS = {  # metadata flag -> question (written after the AI2-THOR action names)
    "pickupable": "Can the robot pick up the object and carry it in its hand (PickupObject)?",
    "openable": "Can the robot open and close the object (OpenObject, CloseObject)?",
    "toggleable": "Can the robot switch the object itself on and off (ToggleObjectOn, ToggleObjectOff)?",
    "sliceable": "Can the robot slice the object into pieces (SliceObject)?",
    "breakable": "Can the object be broken (BreakObject, or by enough force)?",
    "dirtyable": "Can the object be made dirty and clean again (DirtyObject, CleanObject)?",
    "canFillWithLiquid": "Can the robot fill the object with a liquid such as water, coffee or wine (FillObjectWithLiquid)?",
    "receptacle": "Can other objects be put in or on the object (is it a receptacle for PutObject)?",
}
# type/flag pairs where AI2-THOR's own object-type documentation disagrees with the scene metadata: left out
DISPUTED = {("Desktop", "toggleable"), ("Desktop", "breakable"), ("StoveBurner", "toggleable"), ("DogBed", "receptacle"),
            ("Floor", "receptacle"), ("Footstool", "receptacle"), ("Stool", "receptacle"), ("Footstool", "pickupable"),
            ("LaundryHamper", "openable")}
THOR_INTRO = ("AI2-THOR is a simulator of household rooms in which a robot acts on objects with actions such as PickupObject, "
              "OpenObject, ToggleObjectOn, SliceObject and PutObject; each kind of object allows only some of them.")


@functools.lru_cache(maxsize=1)
def thor_properties() -> dict[str, dict[str, set]]:
    """object type -> flag -> set of values seen over all object instances in all scenes."""
    with open(fetch(PROCTHOR, "procthor_ai2thor-object-metadata.json"), encoding="utf-8") as f:
        data = json.load(f)
    values: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    for scenes in data.values():
        for scene in scenes:
            for obj in scene:
                for flag in THOR_ACTIONS:
                    values[obj["objectType"]][flag].add(bool(obj.get(flag)))
    return {t: dict(v) for t, v in values.items()}


@functools.lru_cache(maxsize=1)
def thor_placement() -> dict[str, list[str]]:
    """object type -> receptacle types it may be placed in or on (ReceptacleRestrictions.PlacementRestrictions)."""
    with open(fetch(SIMOBJTYPE, "ai2thor_SimObjType.cs"), encoding="utf-8") as f:
        code = f.read()
    block = code[code.index("PlacementRestrictions = new Dictionary"):]
    block = block[: block.index("};")]
    return {obj: re.findall(r"SimObjType\.(\w+)", body)
            for obj, body in re.findall(r"\{\s*SimObjType\.(\w+),\s*new List<SimObjType>\(\)\s*\{([^}]*)\}", block)}


@register_task
class AI2ThorAffordance(TaskAdapter):
    name = f"{LEAF}/ai2thor_objects"
    leaf = LEAF
    source = "AI2-THOR object metadata (ProcTHOR's dump of 8,715 objects in 195 scenes) and placement restrictions (SimObjType.cs)"
    url = "https://github.com/allenai/ai2thor"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written after AI2-THOR's action and property names; gold from the simulator's per-object flags and placement table"
    sampling = "balanced"
    notes = ("Two kinds of question: whether an object type supports an action (eight flags, answered only when every instance "
             "of the type agrees and AI2-THOR's object-type documentation does not disagree), and whether an object type may be "
             "placed in or on a receptacle type (the simulator's hard-coded placement restrictions; the receptacles offered are "
             "those that appear in the table). Equal shares of yes and no for each kind of question.")

    def load(self):
        rows = []
        for otype, flags in sorted(thor_properties().items()):
            for flag, seen in flags.items():
                if len(seen) == 1 and (otype, flag) not in DISPUTED:
                    rows.append({"kind": flag, "object": otype, "label": "true" if True in seen else "false"})
        table = thor_placement()
        receptacles = sorted({r for rs in table.values() for r in rs})
        for otype, allowed in sorted(table.items()):
            for rec in receptacles:
                if rec != otype:
                    rows.append({"kind": "placement", "object": otype, "receptacle": rec, "label": "true" if rec in allowed else "false"})
        return rows

    def label_of(self, record):
        return f"{record['kind']}|{record['label']}"

    def to_item(self, record, rng):
        obj = f"{record['object']} ({spaced(record['object'])})"
        if record["kind"] == "placement":
            rec = f"{record['receptacle']} ({spaced(record['receptacle'])})"
            text = f"{THOR_INTRO} Does AI2-THOR allow the robot to put the object in or on the receptacle? Assume the receptacle is open and has room."
            state = {"object": obj, "receptacle": rec}
        else:
            text = f"{THOR_INTRO} " + THOR_ACTIONS[record["kind"]]
            state = {"object": obj}
        return self.item(state, {"allowed": Question(type="boolean", instructions=text)}, {"allowed": Gold(label=record["label"])},
                         kind=record["kind"])


# ---------- VirtualHome ----------

VH = "https://raw.githubusercontent.com/xavierpuigf/virtualhome/58970fd80951c2eaa1af713e0917d1a105353ad8/virtualhome/"
VH_ACTIONS = {  # action -> (question, rule on the object's properties and class name), as checked in execution.py
    "grab": ("grab the object (pick it up)", lambda p, c: "GRABBABLE" in p or c in ("water", "child")),
    "open": ("open the object", lambda p, c: "CAN_OPEN" in p or c in ("desk", "window")),
    "switch_on": ("switch on the object", lambda p, c: "HAS_SWITCH" in p),
    "sit": ("sit on the object", lambda p, c: "SITTABLE" in p),
    "lie": ("lie on the object", lambda p, c: "LIEABLE" in p),
    "read": ("read the object", lambda p, c: "READABLE" in p),
    "watch": ("watch the object", lambda p, c: "LOOKABLE" in p),
    "put_on": ("put on the object (wear it)", lambda p, c: "CLOTHES" in p),
    "drink": ("drink from the object", lambda p, c: "DRINKABLE" in p or "RECIPIENT" in p),
    "plug_in": ("plug in the object", lambda p, c: "HAS_PLUG" in p),
    "cut": ("cut the object (holding a knife)", lambda p, c: "EATABLE" in p and "CUTTABLE" in p),
    "type": ("type on the object", lambda p, c: "HAS_SWITCH" in p),
}
VH_INTRO = ("VirtualHome is a simulator of household activities in which an agent performs actions such as grab, open, switch "
            "on, sit, read or cut on objects; each action is possible only on objects whose type supports it. Ignore where the "
            "agent is and what it holds, and judge only whether this kind of object supports the action.")


@register_task
class VirtualHomeAffordance(TaskAdapter):
    name = f"{LEAF}/virtualhome_objects"
    leaf = LEAF
    source = "VirtualHome object properties (properties_data.json) and the action checks of its evolving-graph simulator (Puig et al., 2018)"
    url = "https://github.com/xavierpuigf/virtualhome"
    license = "MIT"
    tier = "C"
    criteria_source = "Written after VirtualHome's action names; gold applies the executor's property check to the object's properties"
    sampling = "balanced"
    notes = ("314 object classes and 12 actions whose executor checks a property of the object (e.g. sit needs SITTABLE, drink "
             "needs DRINKABLE or RECIPIENT, cut needs EATABLE and CUTTABLE); body parts and people are left out. Equal shares of "
             "yes and no for each action.")

    def load(self):
        with open(fetch(VH + "resources/properties_data.json", "virtualhome_properties_data.json"), encoding="utf-8") as f:
            props = json.load(f)
        rows = []
        for cls, plist in sorted(props.items()):
            if "BODY_PART" in plist or "PERSON" in plist:
                continue
            for action, (_, rule) in VH_ACTIONS.items():
                rows.append({"action": action, "object": cls, "label": "true" if rule(set(plist), cls) else "false"})
        return rows

    def label_of(self, record):
        return f"{record['action']}|{record['label']}"

    def to_item(self, record, rng):
        obj = spaced(record["object"])
        text = f"{VH_INTRO} Can the agent {VH_ACTIONS[record['action']][0]}?"
        return self.item({"object": obj}, {"allowed": Question(type="boolean", instructions=text)}, {"allowed": Gold(label=record["label"])},
                         action=record["action"], object_class=record["object"])
