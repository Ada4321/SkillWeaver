import random

from utils.random_control import timestamp_seed_scope


PLACE_TO_TGT_PREFIXES = ("right of ", "left of ")


def _is_place_to_tgt(tgt) -> bool:
    return isinstance(tgt, str) and tgt.startswith(PLACE_TO_TGT_PREFIXES)


class BaseTaskTemplate:
    def __init__(self, surface_tgt_list, force_upright: bool = False, upright_random_prob: float = 0.5):
        self.pickup_template = PickupTaskTemplate()
        self.place_template = PlaceTaskTemplate()
        self.flip_template = FlipTaskTemplate()
        self.straighten_template = StraightenTaskTemplate()
        self.pour_template = PourTaskTemplate()
        self.open_drawer_template = OpenDrawerTaskTemplate()
        self.close_drawer_template = CloseDrawerTaskTemplate()
        self.close_door_template = CloseDoorTaskTemplate()
        self.template_repo = self.make_template()

        self.surface_tgt_list = surface_tgt_list
        self.force_upright = force_upright
        self.upright_random_prob = float(upright_random_prob)

    def make_template(self):
        pickup_templates = self.pickup_template.make_template()
        place_templates_on, place_templates_in, place_templates_to = self.place_template.make_template()
        flip_templates = self.flip_template.make_template()
        straighten_templates = self.straighten_template.make_template()
        pour_templates = self.pour_template.make_template()
        return {
            "pick_up": pickup_templates,
            "place_on": place_templates_on,
            "place_in": place_templates_in,
            "place_to": place_templates_to,
            "flip": flip_templates,
            "straighten": straighten_templates,
            "pour": pour_templates,
            "open_drawer": self.open_drawer_template.make_template(),
            "close_drawer": self.close_drawer_template.make_template(),
            "close_door": self.close_door_template.make_template(),
            "close_door_desc": self.close_door_template.make_template_desc(),
        }

    def sample(self, task: tuple, ):
        task_type = task[0]
        if task_type == "seq":
            if len(task) < 2:
                raise ValueError(f"Sequence task must be ['seq', steps], got: {task!r}")
            steps = list(task[1])
            if not isinstance(steps, list) or len(steps) == 0:
                raise ValueError(f"Sequence task steps must be a non-empty list, got: {steps!r}")
            if len(task) >= 3 and task[2] == 1:
                with timestamp_seed_scope():
                    random.shuffle(steps)
            commands = [self._sample_primitive(step) for step in steps]
            return self._compose_sequence_command(commands)

        return self._sample_primitive(task)

    def _sample_primitive(self, task: tuple, ):
        task_type = task[0]
        # open/close drawer: ["open_drawer"|"close_drawer", "<drawer_descriptor>", "<cabinet>"]
        # -> "Open the <descriptor> drawer of the <cabinet>." (explicit; not the generic src/tgt path).
        if task_type in ("open_drawer", "close_drawer"):
            templates = self.template_repo[task_type]
            with timestamp_seed_scope():
                template = random.choice(templates)
            return template.format(drawer=task[1], cabinet=task[2])
        # close_door: ["close_door", "<object>"] -> "Close the <object> door." (single-door asset);
        # ["close_door", "<door_descriptor>", "<object>"] names the door like the drawer entries.
        if task_type == "close_door":
            if len(task) >= 3:
                templates = self.template_repo["close_door_desc"]
                with timestamp_seed_scope():
                    template = random.choice(templates)
                return template.format(door=task[1], object=task[2])
            templates = self.template_repo[task_type]
            with timestamp_seed_scope():
                template = random.choice(templates)
            return template.format(object=task[1])
        if task_type == "place":
            tgt = task[-2]
            if _is_place_to_tgt(tgt):
                task_type = "place_to"
            elif tgt in self.surface_tgt_list:
                task_type = "place_on"
            else:
                task_type = "place_in"

        templates = self.template_repo[task_type]
        with timestamp_seed_scope():
            template = random.choice(templates)

        if len(task) <= 3:
            command = template.format(src=task[1])
        else:
            command = template.format(src=task[2], tgt=task[3])
            status = task[-1]
            should_append_upright = False
            if self.force_upright:
                if status == 1:
                    should_append_upright = True
                elif status == 2:
                    with timestamp_seed_scope():
                        should_append_upright = random.random() < self.upright_random_prob
            if should_append_upright:
                command = command.replace(".", "UPRIGHT.")
        return command

    @staticmethod
    def _compose_sequence_command(commands: list[str]) -> str:
        clean = [c.strip() for c in commands if isinstance(c, str) and c.strip()]
        if not clean:
            raise ValueError("Sequence task generated no valid commands.")
        if len(clean) == 1:
            return clean[0]

        def _strip_period(s: str) -> str:
            return s[:-1] if s.endswith(".") else s

        parts: list[str] = []
        len_clean = len(clean)
        for idx, cmd in enumerate(clean):
            body = _strip_period(cmd)
            if idx == 0:
                parts.append(f"First, {body}.")
            elif idx == len_clean - 1 and len_clean > 2:
                parts.append(f"Finally, {body}.")
            else:
                parts.append(f"Then, {body}.")
        return " ".join(parts)



class PickupTaskTemplate:
    # verbs = ["Pick up", "Grasp", "Grab", "Lift", "Collect", "Get", "Retrieve", "Take", "Seize", "Fetch", "Get hold of"]
    verbs = ["Pick up"]
    init_locations = [
        "", 
        # "on the table", "located on the table", "off the table surface", "from the table",
        # "on the desktop", "located on the desktop", "off the desktop surface", "from the desktop",
        # "on the surface", "located on the surface", "off the surface", "from the surface",
        # "on the table, and put it back down", "located on the table, and put it back down", "off the table surface, and put it back down", "from the table, and put it back down",
        # "on the desktop, and put it back down", "located on the desktop, and put it back down", "off the desktop surface, and put it back down", "from the desktop, and put it back down",
        # "on the surface, and put it back down", "located on the surface, and put it back down", "off the surface, and put it back down", "from the surface, and put it back down"
    ]

    def make_template(self):
        templates = []
        for verb in self.verbs:
            for location in self.init_locations:
                template = verb + " the {src} " + location + "."
                templates.append(template)
        return templates


class PlaceTaskTemplate:
    # verbs = ["Place", "Put", "Lay", "Position", "Situate", "Set down", "Deposit", "Rest", "Settle"]
    verbs = ["Place", "Put", "Position", "Situate", "Set down", "Deposit", "Rest", "Settle"]
    verbs_to = ["Move", "Transfer", "Relocate", "Shift", "Carry"]

    def make_template(self):
        on_templates = []
        for verb in self.verbs:
            on_templates.append(verb + " the {src} on the {tgt}.")
            on_templates.append(verb + " the {src} onto the {tgt}.")
        in_templates = []
        for verb in self.verbs:
            in_templates.append(verb + " the {src} in the {tgt}.")
            in_templates.append(verb + " the {src} into the {tgt}.")
            in_templates.append(verb + " the {src} inside the {tgt}.")
        to_templates = []
        # "to / over to the {tgt}" phrasings are ambiguous about containment (e.g.
        # "Move the mug to the microwave" doesn't say INSIDE), so they are excluded
        # from the place_on/place_in pools.
        # for verb in self.verbs_to:
        #     to_templates.append(verb + " the {src} to the {tgt}.")
        #     to_templates.append(verb + " the {src} over to the {tgt}.")
        place_to_templates = []
        for verb in ["Place", "Put"]:
            place_to_templates.append(verb + " the {src} to the {tgt}.")
        return on_templates + to_templates, in_templates + to_templates, place_to_templates


class FlipTaskTemplate:
    verbs = ["Flip", "Turn", "Rotate", "Invert", "Toss", "Spin", "Revolve", "Swivel", "Whirl"]
    orientations = ["over", "upside down", "around", "180 degrees, making it upside down", "and make it face down", "and make its opening face down"]

    def make_template(self):
        templates = []
        for verb in self.verbs:
            for orientation in self.orientations:
                template = verb + " the {src} " + orientation + "."
                templates.append(template)
        return templates


class StraightenTaskTemplate:
    def make_template(self):
        return [
            "Straighten up the {src}.",
            "Upright the {src}.",
            "Set the {src} upright.",
            "Make the {src} upright.",
            "Stand up the {src}.",
            "Place the {src} upright.",
            "Position the {src} upright.",
            "Make the {src} stand up.",
            "Make the {src} stand upright.",
            "Make the {src} stand straight up.",
            "Straighten up the lying {src}.",
            "Upright the lying {src}.",
            "Set the lying {src} upright.",
            "Make the lying {src} upright.",
            "Stand up the lying {src}.",
            "Place the lying {src} upright.",
            "Position the lying {src} upright.",
            "Make the lying {src} stand up.",
            "Make the lying {src} stand upright.",
            "Make the lying {src} stand straight up."
        ]


class PourTaskTemplate:
    def make_template(self):
        return [
            # "Tilt the {src} to pour its contents into the {tgt}.",
            # "Empty the contents of the {src} into the {tgt}.",
            # "Transfer the liquid from the {src} to the {tgt}.",
            # "Dump the contents of the {src} into the {tgt}.",
            # "Decant the contents of the {src} into the {tgt}.",
            # "Tip the {src} to pour into the {tgt}.",
            "Pour from the {src} into the {tgt}.",
            # "Tilt the {src} to pour its contents into the {tgt} and put the {src} down.",
            # "Empty the contents of the {src} into the {tgt} and put the {src} down.",
            # "Transfer the liquid from the {src} to the {tgt} and put the {src} down.",
            # "Dump the contents of the {src} into the {tgt} and put the {src} down.",
            # "Decant the contents of the {src} into the {tgt} and put the {src} down.",
            # "Tip the {src} to pour into the {tgt} and put the {src} down.",
            # "Pour from the {src} into the {tgt} and put the {src} down."
        ]


class OpenDrawerTaskTemplate:
    # Multi-DOF cabinet: a drawer is named by descriptor (top/middle/bottom), the cabinet by name.
    # Entry form: ["open_drawer", "<drawer_descriptor>", "<cabinet>"]. See docs/arti/task_instruction.md.
    verbs = ["Open", "Pull open", "Slide open"]

    def make_template(self):
        return [verb + " the {drawer} drawer of the {cabinet}." for verb in self.verbs]


class CloseDrawerTaskTemplate:
    verbs = ["Close", "Push", "Shut", "Slide shut"]

    def make_template(self):
        templates = ["Close the {drawer} drawer of the {cabinet}.",
                     "Push the {drawer} drawer of the {cabinet} shut.",
                     "Shut the {drawer} drawer of the {cabinet}.",
                     "Push in the {drawer} drawer of the {cabinet}."]
        return templates


class CloseDoorTaskTemplate:
    # Revolute door(s), e.g. microwave. Entry forms: ["close_door", "<object>"] (single-door
    # asset) or ["close_door", "<door_descriptor>", "<object>"] (multi-door; the descriptor
    # names the door, mirroring the drawer entries).
    def make_template(self):
        return ["Close the {object} door.",
                "Push the {object} door shut.",
                "Shut the {object} door.",
                "Close the door of the {object}."]

    def make_template_desc(self):
        return ["Close the {door} door of the {object}.",
                "Push the {door} door of the {object} shut.",
                "Shut the {door} door of the {object}."]
