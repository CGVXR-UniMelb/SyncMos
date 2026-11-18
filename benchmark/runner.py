import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from omegaconf import OmegaConf
from tqdm import tqdm

from src.action_planning.event_parser import get_event_parser
from src.action_planning.event_parser.position_sampler import PositionSampler
from src.action_planning.narrator import get_narrator
from src.action_planning.scene_describer.scene_describer import SceneDescriber
from src.scene.scene import Scene
from src.type import Character, Event
from src.utils import misc
from src.utils.misc import CustomEncoder

from .evaluator import Evaluator
from .evaluator import CharacterNotFoundError

_TESTCASE_DIR = Path("./benchmark/test_cases")
_GROUP_DIRS = {
    range(0, 5): _TESTCASE_DIR / "0_OA",
    range(10, 15): _TESTCASE_DIR / "1_RC",
    range(20, 25): _TESTCASE_DIR / "2_SS",
    range(30, 35): _TESTCASE_DIR / "3_OA_RC",
    range(40, 45): _TESTCASE_DIR / "4_OA_SS",
    range(50, 55): _TESTCASE_DIR / "5_RC_SS",
    range(60, 70): _TESTCASE_DIR / "6_PI",
    range(902, 921): _TESTCASE_DIR / "narrator_comparison",  # Dependency test case 902-920
}


def get_test_case_ids() -> list:
    """Returns a list of all test case IDs."""
    ids = []
    for range_ in _GROUP_DIRS:
        ids.extend(range_)
    return ids


def load_test_case(id_: int) -> dict:
    for range_, group_dir in _GROUP_DIRS.items():
        if id_ in range_:
            return json.loads((group_dir / f"testcase_{id_}.json").read_text())
    raise ValueError(f"Test case with ID {id_} not found.")


@dataclass
class TestMetadata:
    test_case_id: int
    distraction_level: int
    repeat_idx: int

    @classmethod
    def from_str(cls, metadata_str: str):
        m = re.match(r"tc(\d+)-d(\d+)-r(\d+)", metadata_str)
        if not m:
            raise ValueError(f"Invalid metadata string: {metadata_str}")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def __repr__(self):
        return f"tc{str(self.test_case_id).zfill(2)}-d{self.distraction_level}-r{self.repeat_idx}"

    def __hash__(self):
        return hash((self.test_case_id, self.distraction_level, self.repeat_idx))


class BenchmarkRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.evaluator = Evaluator()
        self.test_case_ids = cfg.evaluation.test_cases
        self.repeat_count = cfg.evaluation.repeat_count
        self.distraction_levels = cfg.evaluation.distraction_levels
        self.scene_description_type = cfg.scene_describer.description_type
        self.planning_mode = "event" if not cfg.narrator.disable_event_based_planning else "no_event"
        if cfg.event_parser.disable_lcps:
            self.planning_mode += "_no_lcps"

        self.run_id = datetime.now().strftime("%y%m%d-%H%M%S")
        self.logger = misc.get_console_logger("BenchmarkRunner", cfg.logging_level)
        self.log_base_dir = Path(
            f"./logs/benchmark/{cfg.evaluation.experiment_name}/"
            f"{self.run_id}-{self.scene_description_type}-{self.planning_mode}"
        )
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, self.log_base_dir / "config.yaml", resolve=True)

    def run_tests(self):
        test_case_ids_by_scene = self._group_test_cases_by_scene()
        for scene_name, test_case_ids in tqdm(test_case_ids_by_scene.items(), desc="Scenes"):
            if not test_case_ids:
                continue

            self._setup_scene(scene_name)
            for id_ in tqdm(test_case_ids, desc="Test cases"):
                test_case = load_test_case(id_)
                for distraction_level in tqdm(self.distraction_levels, desc="Distraction levels"):
                    max_distraction_level = self._get_max_distraction_level(test_case)
                    if distraction_level > max_distraction_level:
                        continue
                    for repeat_idx in tqdm(range(self.repeat_count), desc="Repeats", leave=False):
                        self.run_test(test_case, distraction_level, repeat_idx)

    def run_test(self, test_case, distraction_level, repeat_idx):
        self._setup_modules_and_scenario(test_case, distraction_level, repeat_idx)

        tags = test_case["tags"]
        metadata = TestMetadata(test_case["id"], distraction_level, repeat_idx)
        try:
            parsed_event, prompt_tokens, completion_tokens = self._run_planning()
            if "PI" in tags and not self.planning_mode.endswith("no_lcps"):
                for parsed_plan in parsed_event.values():
                    parsed_plan["sampled_position"] = self._sample_position(parsed_plan)

            # Choose the evaluation method based on the parser type
            from src.action_planning.event_parser.batch_event_parser import BatchEventParser

            if isinstance(self.event_parser, BatchEventParser):
                # Grid method: Only evaluate actions and positions
                passed = self._evaluate_grid_result(test_case, parsed_event)
            else:
                # Traditional method: Evaluate actions and relationships
                passed = self.evaluator.evaluate_result(test_case, parsed_event)
            self._save_result(test_case, metadata, parsed_event, prompt_tokens, completion_tokens, passed)
        except Exception as e:
            self.logger.warning(f"Error occurred during: {metadata}")
            self._save_error_logs(metadata, e, traceback.format_exc())
            self._save_error_result(tags, metadata, e)

    def _run_planning(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0
        event, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
            self.scene_description,
            self.available_action_labels,
            self.user_instruction,
            with_usage=True,
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        if event is None:  # when LLM wants to skip the turn
            return {}, total_prompt_tokens, total_completion_tokens
        if not self.narrator.event_based_planning:
            parsed_event = event  # already parsed
            return parsed_event, total_prompt_tokens, total_completion_tokens

        # Check if it’s a batch parser
        from src.action_planning.event_parser.batch_event_parser import BatchEventParser

        if isinstance(self.event_parser, BatchEventParser):
            # Batch parser processes the event list
            events = [event] if not isinstance(event, list) else event
            parsed_events, prompt_tokens, completion_tokens = self.event_parser.parse_events(
                events,
                self.available_action_labels,
                self.scene_description,
                with_usage=True,
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            # Convert to a single event format (format expected by benchmark)
            if parsed_events and 0 in parsed_events:
                parsed_event = parsed_events[0]
            else:
                parsed_event = {}
        else:
            # Traditional parser
            parsed_event, prompt_tokens, completion_tokens = self.event_parser.parse_event(
                event,
                self.scene_description,
                self.available_action_labels,
                self.narrator.previous_events[:-1],  # since the last event is the current event
                with_usage=True,
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        return parsed_event, total_prompt_tokens, total_completion_tokens

    def _group_test_cases_by_scene(self):
        test_case_ids = {"House": [], "Office": [], "Restaurant": []}
        for id_ in self.test_case_ids:
            test_case = load_test_case(id_)
            scene_name = test_case["scene_name"]
            test_case_ids[scene_name].append(id_)
        return test_case_ids

    def _setup_scene(self, scene_name):
        self.scene = Scene(scene_name, self.cfg)
        self.scene_name = self.scene.scene_name
        self.scene_mesh = self.scene.scene_mesh
        self.objects = self.scene.objects
        self.grid_map = self.scene.grid_map
        self.scene_graph = self.scene.scene_graph
        # Build a fast object finder for relationship checks
        self.object_finder = {f"{obj.label}_{obj.idx}": obj for obj in self.objects}

    def _get_object_color(self, obj):
        """Get color based on object type"""
        color_map = {
            "chair": "#8B4513",  # brown
            "table": "#D2691E",  # orange
            "cabinet": "#A0522D",  # dark brown
            "sofa": "#CD853F",  # light brown
            "bed": "#DEB887",  # beige
            "desk": "#F4A460",  # sandy brown
            "shelf": "#BC8F8F",  # rosy brown
            "counter": "#D2B48C",  # light brown
            "stool": "#8B7355",  # dark brown
            "bench": "#A0522D",  # dark brown
        }

        for obj_type, color in color_map.items():
            if obj_type in obj.label.lower():
                return color

        return "#808080"  # default gray

    def _draw_object(self, ax, obj, alpha=0.7):
        """Draw object"""
        center = obj.center[:2]
        width = obj.width
        depth = obj.depth
        orientation = obj.orientation

        # Calculate rotated four corner points
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)

        # Unrotated four corner points (relative to center)
        corners = np.array(
            [[-width / 2, -depth / 2], [width / 2, -depth / 2], [width / 2, depth / 2], [-width / 2, depth / 2]]
        )

        # Rotate
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation_matrix.T

        # Translate to actual position
        final_corners = rotated_corners + center

        # Ensure correct shape (4, 2)
        if final_corners.shape != (4, 2):
            # Use simple rectangle if shape is wrong
            final_corners = np.array(
                [
                    [center[0] - width / 2, center[1] - depth / 2],
                    [center[0] + width / 2, center[1] - depth / 2],
                    [center[0] + width / 2, center[1] + depth / 2],
                    [center[0] - width / 2, center[1] + depth / 2],
                ]
            )

        # Draw object
        color = self._get_object_color(obj)
        rect = patches.Polygon(final_corners, closed=True, facecolor=color, edgecolor="black", alpha=alpha, linewidth=1)
        ax.add_patch(rect)

        # Add label
        ax.text(
            center[0],
            center[1],
            f"{obj.label}_{obj.idx}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
        )

    def _draw_position(self, ax, position, label, color="red", size=100):
        """Draw position point"""
        if len(position) >= 2:
            x, y = position[0], position[1]
        else:
            return

        ax.scatter(x, y, c=color, s=size, marker="*", edgecolors="black", linewidth=1, zorder=10)
        ax.text(
            x,
            y + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    def _generate_visualization(self, test_case, planning_result, metadata):
        """Generate visualization for test case result"""
        try:
            # Create visualization directory
            vis_dir = (
                self.log_base_dir
                / "visualizations"
                / f"testcase-{metadata.test_case_id}"
                / f"distraction-{metadata.distraction_level}"
            )
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # Draw scene objects
            for obj in self.objects:
                self._draw_object(ax, obj)

            # Draw valid position range
            self._draw_valid_position_range(ax, test_case)

            # Draw expected positions
            for i, expected_plan in enumerate(test_case["expected_plans"]):
                if "sampled_position" in expected_plan:
                    pos = expected_plan["sampled_position"]
                    self._draw_position(ax, pos, f"Expected {i+1}", "green", 150)

            # Draw generated positions
            for char_name, char_plan in planning_result.items():
                # Handle both BatchEventParser (target_position) and regular parser (sampled_position)
                if "target_position" in char_plan:
                    pos = char_plan["target_position"]
                    self._draw_position(ax, pos, f"Generated {char_name}", "red", 150)
                elif "sampled_position" in char_plan:
                    pos = char_plan["sampled_position"]
                    self._draw_position(ax, pos, f"Generated {char_name}", "red", 150)
                elif "relationships" in char_plan and char_plan["relationships"]:
                    # For Area method: try to infer position from relationships
                    inferred_pos = self._infer_position_from_relationships(char_plan["relationships"])
                    if inferred_pos is not None:
                        self._draw_position(ax, inferred_pos, f"Generated {char_name}", "red", 150)

            # Add event text in top-left corner
            event_text = test_case.get("user_instruction", "No event description")
            # Wrap long text to multiple lines
            max_chars_per_line = 50
            wrapped_text = []
            words = event_text.split()
            current_line = ""

            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += (" " + word) if current_line else word
                else:
                    if current_line:
                        wrapped_text.append(current_line)
                    current_line = word

            if current_line:
                wrapped_text.append(current_line)

            event_display = "\n".join(wrapped_text)

            ax.text(
                0.02,
                0.98,
                f"Event:\n{event_display}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="black"),
                zorder=1000,
            )

            # Set title and labels
            ax.set_title(
                f'Test Case {metadata.test_case_id} - {test_case["scene_name"]} Scene\nDistraction: {metadata.distraction_level}, Repeat: {metadata.repeat_idx}',
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

            # Add legend
            legend_elements = [
                plt.Line2D(
                    [0], [0], marker="*", color="green", linestyle="None", markersize=10, label="Expected Position"
                ),
                plt.Line2D(
                    [0], [0], marker="*", color="red", linestyle="None", markersize=10, label="Generated Position"
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

            # Save figure
            vis_file = vis_dir / f"{metadata}.png"
            plt.savefig(vis_file, dpi=300, bbox_inches="tight")
            plt.close()

            return str(vis_file)
        except Exception as e:
            print(f"⚠️  Failed to generate visualization: {e}")
            return None

    def _infer_position_from_relationships(self, relationships):
        """Infer position from relationships for Area method visualization"""
        try:
            if not relationships:
                return None

            # Use the first relationship to infer position
            relationship = relationships[0]

            # Parse relationship like "close to table_3" or "adjacent to chair_1"
            parts = relationship.split()
            if len(parts) >= 3:
                predicate = " ".join(parts[:-1])
                anchor_name = parts[-1]

                # Find the anchor object
                anchor_obj = None
                for obj in self.objects:
                    if f"{obj.label}_{obj.idx}" == anchor_name:
                        anchor_obj = obj
                        break

                if anchor_obj is None:
                    return None

                # Infer position based on relationship type
                if predicate in ["close to", "adjacent to"]:
                    # Place near the object
                    offset = 1.0  # 1 meter offset
                    angle = 0  # Default angle

                    # Add some randomness to avoid overlapping
                    import random

                    angle = random.uniform(0, 2 * 3.14159)

                    x = anchor_obj.center[0] + offset * np.cos(angle)
                    y = anchor_obj.center[1] + offset * np.sin(angle)
                    return [x, y]

                elif predicate in ["between"]:
                    # For "between A and B", place at midpoint
                    # This is a simplified version - would need more complex parsing
                    return [anchor_obj.center[0], anchor_obj.center[1]]

                elif predicate in ["sit on", "lie on"]:
                    # Place on the object (for chairs, beds, etc.)
                    return [anchor_obj.center[0], anchor_obj.center[1]]

            return None

        except Exception as e:
            print(f"⚠️  Failed to infer position from relationships: {e}")
            return None

    def _draw_valid_position_range(self, ax, test_case):
        """Draw the valid position range"""
        try:
            import numpy as np

            # Obtain the relationship conditions
            relationships = test_case["expected_plans"][0]["relationships"]
            if not relationships:
                return

            # Calculate the scene boundaries
            all_x = [obj.center[0] for obj in self.objects]
            all_y = [obj.center[1] for obj in self.objects]
            margin = 1.0
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin

            # Create a grid to test each position
            resolution = 0.2  # Reduce the resolution for faster processing
            x_coords = np.arange(x_min, x_max, resolution)
            y_coords = np.arange(y_min, y_max, resolution)
            X, Y = np.meshgrid(x_coords, y_coords)

            # Test each position
            valid_positions = np.zeros_like(X, dtype=bool)

            for i in range(len(y_coords)):
                for j in range(len(x_coords)):
                    position = [X[i, j], Y[i, j]]
                    try:
                        passed, error = self.evaluator.position_evaluator.evaluate(relationships, position)
                        valid_positions[i, j] = passed
                    except:
                        valid_positions[i, j] = False

            # Draw the valid position range
            ax.contourf(X, Y, valid_positions.astype(float), levels=[0.5, 1.5], colors=["red"], alpha=0.2)

        except Exception as e:
            print(f"⚠️  Failed to draw valid position range: {e}")

    def _setup_modules_and_scenario(self, test_case, distraction_level, repeat_idx):
        self.user_instruction = test_case["user_instruction"]
        self.available_action_labels = test_case["action_labels"]

        # ----- setup characters and their current states -----
        self.characters: List[Character] = []
        for character in test_case["current_states"]:
            if character["type"] == "distractor" and character["distraction_level"] > distraction_level:
                continue
            character_ = Character(
                name=character["name"],
                state=character["state"],
                current_actions=[character["current_action"]],
                relationships=character["position"].copy() + character["orientation"].copy(),
            )
            if any(rel.startswith("sit on") for rel in character_.relationships):
                character_.current_actions.append("sit")
            if any(rel.startswith("lie on") for rel in character_.relationships):
                character_.current_actions.append("lie")
            self.characters.append(character_)

        # ----- setup scene description -----
        self.scene_describer = SceneDescriber(self.cfg)
        if self.scene_description_type in ["narrative", "narrative_vlm"]:
            self.scene_description = self.scene_describer.get_pregenerated_scene_description(
                self.scene_name, idx=repeat_idx
            )
        else:
            self.scene_description = self.scene_describer.get_scene_description(self.scene_name, self.scene_graph)

        # ----- setup narrator -----
        self.narrator = get_narrator(self.characters, self.objects, self.cfg)
        self.narrator.available_characters = [c for c in self.characters if c.state == "idle"]
        if test_case["allow_null_plan"]:
            self.narrator.additional_guidelines += "- You can skip the event generation by outputting an empty event that involves no characters in it."  # fmt:skip
            self.narrator.additional_guidelines += " This is useful when you need to wait for some characters to become idle before the next event."  # fmt:skip
            if self.planning_mode.startswith("no_event"):
                self.narrator.additional_guidelines = self.narrator.additional_guidelines.replace("event", "plan")
        self.narrator.log_id_prefix = f"repeat-{repeat_idx}"
        self.narrator.log_basedir = (
            self.log_base_dir / "narrator" / f"testcase-{test_case['id']}" / f"distraction-{distraction_level}"
        )
        for example_file in self.narrator.example_files[:]:
            if example_file.stem not in test_case["tags"]:
                self.narrator.example_files.remove(example_file)

        # ----- setup planning history (previous events) -----
        for plan in test_case["planning_history"]:
            if plan["type"] == "distractor" and plan["distraction_level"] > distraction_level:
                continue
            event_ = Event(
                involved_characters=[c for c in self.characters if c.name in plan["characters"]],
                activity=plan["activity"],
                state=plan["state"],
            )
            for c in self.characters:
                if c.name not in plan["characters"]:
                    continue
                if plan["state"] == "completed":
                    c.event = None
                else:
                    c.event = event_  # assign event to character (overwrite previous one)

            parsed_event_ = {}
            for parsed in plan["details"]:
                parsed_event_[parsed["character"]] = {}
                parsed_event_[parsed["character"]]["target_action"] = [parsed["target_action"]]
                parsed_event_[parsed["character"]]["relationships"] = (
                    parsed["position"].copy() + parsed["orientation"].copy()
                )
            event_.parsed_event = parsed_event_
            self.narrator.previous_events.append(event_)

        # ----- setup event parser -----
        self.event_parser = get_event_parser(
            self.characters,
            self.objects,
            self.cfg,
            scene_name=self.scene_name,  # Provide scene_name for the batch parser
            grid_map=self.grid_map,  # Provide grid_map for the batch parser
        )
        self.event_parser.log_id_prefix = f"repeat-{repeat_idx}"
        self.event_parser.log_basedir = (
            self.log_base_dir / "event_parser" / f"testcase-{test_case['id']}" / f"distraction-{distraction_level}"
        )
        # Only the traditional parser has code_executor
        if hasattr(self.event_parser, "code_executor"):
            self.event_parser.code_executor.set_previous_events(self.narrator.previous_events)
        # Only the traditional parser has example_files
        if hasattr(self.event_parser, "example_files"):
            for example_file in self.event_parser.example_files[:]:
                if example_file.stem not in test_case["tags"]:
                    self.event_parser.example_files.remove(example_file)

        # ----- setup position sampler (used for PI tags) -----
        self.position_sampler = PositionSampler(self.grid_map, self.cfg)
        self.position_sampler.update_anchors(self.characters, self.objects)

        # ----- setup evaluators -----
        self.evaluator.setup_evaluator(
            self.available_action_labels,
            self.characters,
            self.objects,
        )

    def _save_result(self, test_case, metadata, planning_result, prompt_tokens, completion_tokens, passed):
        result_dir = (
            self.log_base_dir
            / "results"
            / f"testcase-{metadata.test_case_id}"
            / f"distraction-{metadata.distraction_level}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{metadata}.json"

        # Generate visualization
        vis_file = self._generate_visualization(test_case, planning_result, metadata)

        with open(result_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "scene_name": test_case["scene_name"],
                        "tags": test_case["tags"],
                        "run_id": self.run_id,
                        "scene_description_type": self.scene_description_type,
                        "planning_mode": self.planning_mode,
                        "test_case_id": metadata.test_case_id,
                        "distraction_level": metadata.distraction_level,
                        "repeat_idx": metadata.repeat_idx,
                        "user_instruction": test_case["user_instruction"],
                        "expected_plans": test_case["expected_plans"],
                        "allow_null_plan": test_case["allow_null_plan"],
                        "planning_result": planning_result,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "error_occurred": False,
                        "error_message": None,
                        "passed": passed,
                        "visualization_file": vis_file,
                    },
                    cls=CustomEncoder,
                )
            )

    def _save_error_result(self, tags, metadata, exception):
        result_dir = (
            self.log_base_dir
            / "results"
            / f"testcase-{metadata.test_case_id}"
            / f"distraction-{metadata.distraction_level}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{metadata}.json"
        with open(result_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "tags": tags,
                        "run_id": self.run_id,
                        "scene_description_type": self.scene_description_type,
                        "planning_mode": self.planning_mode,
                        "test_case_id": metadata.test_case_id,
                        "distraction_level": metadata.distraction_level,
                        "repeat_idx": metadata.repeat_idx,
                        "parsed_event": None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "error_occurred": True,
                        "error_message": f"{type(exception).__name__}: {str(exception)}",
                        "passed": False,
                    },
                    cls=CustomEncoder,
                )
            )

    def _save_error_logs(self, metadata, exception, stack_trace):
        error_log_dir = (
            self.log_base_dir
            / "errors"
            / f"testcase-{metadata.test_case_id}"
            / f"distraction-{metadata.distraction_level}"
        )
        error_log_dir.mkdir(parents=True, exist_ok=True)
        error_log_file = error_log_dir / f"{metadata}.txt"
        with open(error_log_file, "w") as f:
            f.write(f"{type(exception).__name__}: {str(exception)}\n")
            f.write(stack_trace)

    def _sample_position(self, parsed_plan):
        conditions = parsed_plan["relationships"]
        conditions = [c.replace("inside", "sit on") for c in conditions]
        conditions = [c.replace("interact with", "adjacent to") for c in conditions]
        sampled_position = (
            self.position_sampler.sample_position(
                conditions,
                positions_to_avoid=[],
                position_to_nearby=None,
                return_tensor=False,
            )
            .squeeze()[:2]
            .tolist()
        )
        return sampled_position

    def _evaluate_grid_result(self, test_case, planning_result):
        """Evaluation using the grid method: Only evaluate actions and positions"""
        if test_case["allow_null_plan"] and not planning_result:
            return True

        passed = []
        for expected_plan in test_case["expected_plans"]:
            try:
                if "PI" in test_case["tags"]:
                    # Position reasoning: Only evaluate positions
                    passed_ = self._evaluate_position_only(expected_plan, planning_result)
                else:
                    # Other tasks: Evaluate actions and positions
                    passed_ = self._evaluate_action_and_position(expected_plan, planning_result)
                passed.append(passed_)
            except Exception as e:
                raise e
        return all(passed)

    def _evaluate_action_and_position(self, expected_plan, planning_result):
        """Evaluate actions and positions (No relationships evaluated)"""
        if expected_plan["character"] not in planning_result.keys():
            raise CharacterNotFoundError(f"{expected_plan['character']} not found in planning result")

        character_plan = planning_result[expected_plan["character"]]

        # 1. Evaluate actions
        action_passed = self.evaluator.action_evaluator.evaluate(
            expected_plan["target_action"],
            character_plan["target_action"],
        )
        if not action_passed:
            print("❌ Action evaluation failed:")
            print(f"Expected action: {expected_plan['target_action']}")
            print(f"Actual action: {character_plan['target_action']}")
            print(f"Available actions: {self.available_action_labels}")
            return False

        # 2. Evaluation position (if there is an expected position)
        if "sampled_position" in expected_plan and "sampled_position" in character_plan:
            position_passed = self._evaluate_position_by_relationship(
                expected_plan["relationships"], character_plan["sampled_position"]
            )
            if not position_passed:
                print("❌ Position evaluation failed:")
                print(f"Relationship condition: {expected_plan['relationships']}")
                print(f"Actual position: {character_plan['sampled_position']}")
            return position_passed

        return True  # If there is no expected position, only actions are evaluated

    def _evaluate_position_only(self, expected_plan, planning_result):
        """Only position evaluation (for PI tasks)"""
        if expected_plan["character"] not in planning_result.keys():
            raise CharacterNotFoundError(f"{expected_plan['character']} not found in planning result")

        character_plan = planning_result[expected_plan["character"]]
        position_passed = self._evaluate_position_by_relationship(
            expected_plan["relationships"], character_plan["sampled_position"]
        )
        if not position_passed:
            print("❌ Position evaluation failed (PI tasks):")
            print(f"Relationship condition: {expected_plan['relationships']}")
            print(f"Actual position: {character_plan['sampled_position']}")
        return position_passed

    def _evaluate_position_by_relationship(self, relationships, actual_position):
        """Evaluating position using relationship conditions - Specific for Grid mode"""
        if not relationships:
            return True

        # Grid mode: Directly checks geometric relationships without parsing predicate strings
        return self._evaluate_grid_position_geometry(relationships, actual_position)

    def _evaluate_grid_position_geometry(self, relationships, actual_position):
        """Grid Mode: Directly checks whether geometric relationships are established"""

        # Analyzes relationship conditions
        conditions = []
        for rel in relationships.split("||"):
            rel = rel.strip()
            if rel:
                conditions.append(rel)

        if not conditions:
            return True

        # Checks whether any condition is met
        for condition in conditions:
            if self._check_single_condition(condition, actual_position):
                return True

        return False

    def _check_single_condition(self, condition, position):
        """Checks individual relationship conditions"""

        # Processing "between A and B" Format
        if "between" in condition and " and " in condition:
            return self._check_between_condition(condition, position)

        # Processes other relationship conditions
        return self._check_standard_condition(condition, position)

    def _check_between_condition(self, condition, position):
        """Conducts checks 'between A and B' Conditions - Uses more scientific geometric judgments"""

        # Parse… "between chair_1 and chair_3"
        parts = condition.split()
        if len(parts) >= 4 and parts[0] == "between" and parts[2] == "and":
            # Cleans up any trailing punctuation, such as right parentheses/Commas
            obj1_name = parts[1].strip(")],, ")
            obj2_name = parts[3].strip(")],, ")

            # Obtains objects
            obj1 = getattr(self, "object_finder", {}).get(obj1_name)
            obj2 = getattr(self, "object_finder", {}).get(obj2_name)

            if obj1 is None or obj2 is None:
                print(f"⚠️ Object not found: {obj1_name} or {obj2_name}")
                return False

            # Using a more intelligent between determination
            return self._is_position_between_objects(position, obj1, obj2)

        return False

    def _is_position_between_objects(self, position, obj1, obj2):
        """A more scientific between determination: checking if the position is on the line between two objects and ensuring the distance is reasonable"""
        import numpy as np

        pos = np.array(position)
        c1 = np.array(obj1.center[:2])
        c2 = np.array(obj2.center[:2])

        # Calculating the line vector
        line_vec = c2 - c1
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-6:  # The two objects are almost overlapping
            return np.linalg.norm(pos - c1) <= 1.0

        # Calculating the projection of the position on the line
        pos_vec = pos - c1
        t = np.dot(pos_vec, line_vec) / (line_length * line_length)

        # Projection point
        projection = c1 + t * line_vec

        # 1. Checking if it is within the range of the line (allowing a certain extent)
        margin = 0.5  # Allow for expansion at both ends of the line 0.5 Meter
        if t < -margin or t > 1 + margin:
            return False

        # 2. Checking the vertical distance to the line
        perpendicular_distance = np.linalg.norm(pos - projection)

        # Dynamically adjusting the threshold based on object size
        avg_size = (obj1.width + obj1.depth + obj2.width + obj2.depth) / 4
        max_perpendicular = max(0.3, avg_size * 0.3)  # At least 0.3 Meters, or 30% of the average size of the object%

        passed = perpendicular_distance <= max_perpendicular

        if not passed:
            print("❌ 'between' Relationship check failed:")
            print(f"Location: {position}")
            print(f"   Object 1: {obj1.label}_{obj1.idx} at {c1}")
            print(f"   Object 2: {obj2.label}_{obj2.idx} at {c2}")
            print(f"Projection parameter t: {t:.3f}")
            print(f"Vertical distance: {perpendicular_distance:.3f} (Threshold: {max_perpendicular:.3f})")

        return passed

    def _check_standard_condition(self, condition, position):
        """Check standard relationship conditions - Uses more scientific geometric judgments"""

        # Parsing conditions "adjacent to cabinet_3"
        parts = condition.split()
        if len(parts) < 2:
            return False

        predicate = " ".join(parts[:-1])
        # Cleans up any trailing punctuation, such as right parentheses/Commas
        anchor_name = parts[-1].strip(")],, ")

        # Obtaining anchor object
        anchor = getattr(self, "object_finder", {}).get(anchor_name)
        if anchor is None:
            print(f"⚠️ Anchor object not found: {anchor_name}")
            return False

        # Using more intelligent relationship determination
        return self._check_relationship_with_object(position, predicate, anchor)

    def _check_relationship_with_object(self, position, predicate, anchor):
        """More scientific relationship determination: Adjust thresholds dynamically based on object size and relationship type"""
        import numpy as np
        from src.scene import relationships as sg

        pos = np.array(position)
        anchor_center = np.array(anchor.center[:2])

        # Adjust thresholds dynamically based on relationship type and object size
        if predicate in ["adjacent to", "close to"]:
            # Calculate reasonable distances based on object size
            object_size = max(anchor.width, anchor.depth)
            base_threshold = max(0.5, object_size * 0.5)  # At least 0.5 Meter, or 50% of object size%

            if predicate == "adjacent to":
                threshold = base_threshold
            else:  # close to
                threshold = base_threshold * 1.5  # Close is more relaxed than adjacent

            # Calculate using boundary distances
            distance = sg.get_distance_from_boundary(pos, anchor, offset=0.0)
            passed = distance <= threshold

            if not passed:
                print(f"❌ ’{predicate}’ relationship check failed:")
                print(f"Location: {position}")
                print(f"Anchor points: {anchor.label} _ {anchor.idx} at {anchor_center}")
                print(f"Object size: {object_size:.3f}")
                print(f"Distance: {distance:.3f} (Threshold: {threshold:.3f})")

            return passed

        elif predicate in ["to the left of", "to the right of", "in front of", "behind"]:
            # Direction relationship: Use stricter thresholds
            distance = sg.get_distance_from_relationship(pos, anchor, predicate, offset=0.0)
            threshold = 0.2  # Direction relationships require more precision
            passed = distance <= threshold

            if not passed:
                print(f"❌ ’{predicate}’ relationship check failed:")
                print(f"Location: {position}")
                print(f"Anchor points: {anchor.label} _ {anchor.idx} at {anchor_center}")
                print(f"Distance: {distance:.3f} (Threshold: {threshold})")

            return passed

        elif predicate == "inside":
            # Internal relationships: Check if within object boundaries
            distance = sg.get_distance_from_boundary(pos, anchor, offset=0.0)
            passed = distance <= 0.0  # Must be within boundaries

            if not passed:
                print(f"❌ ’{predicate}’ relationship check failed:")
                print(f"Location: {position}")
                print(f"Anchor points: {anchor.label} _ {anchor.idx} at {anchor_center}")
                print(f"Distance: {distance:.3f} (Must be within the boundary)")

            return passed

        elif predicate == "sit on":
            # Sitting down: Check if it’s in the chair/Sitting relationships require close proximity
            distance = sg.get_distance_from_boundary(pos, anchor, offset=0.0)
            threshold = 0.3  # Relationship predicates are not supported
            passed = distance <= threshold

            if not passed:
                print(f"❌ ’{predicate}’ relationship check failed:")
                print(f"Location: {position}")
                print(f"Anchor points: {anchor.label} _ {anchor.idx} at {anchor_center}")
                print(f"Distance: {distance:.3f} (Threshold: {threshold})")

            return passed

        else:
            print(f"Unupported relation predicate: {predicate}")
            return False

    def _get_max_distraction_level(self, test_case):
        max_distraction_level = 0
        for plan in test_case["planning_history"]:
            if plan["type"] == "distractor":
                max_distraction_level = max(max_distraction_level, plan["distraction_level"])
        for character in test_case["current_states"]:
            if character["type"] == "distractor":
                max_distraction_level = max(max_distraction_level, character["distraction_level"])
        return max_distraction_level
