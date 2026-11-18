from pathlib import Path
import yaml
from datetime import datetime
from .scene_describer import SceneDescriber


class SceneDescriberTimeline(SceneDescriber):
    def __init__(self, cfg) -> None:
        # Call the parent class’s initialization method
        super().__init__(cfg)

        # Select the example processing method
        self.use_timeline_examples = getattr(cfg.scene_describer, "use_timeline_examples", False)

        # Automatically save the generated description
        self.auto_save_descriptions = getattr(cfg.scene_describer, "auto_save_descriptions", True)

    def _create_example_message(self, example_file):
        """
        Override the example message creation method
        If timeline_examples is enabled, directly use response as the assistant part
        Otherwise, use the parent class’s default behavior
        """
        if self.use_timeline_examples:
            # Timeline mode: Directly use response without generating the user part
            example_file = Path(example_file)
            example_data = yaml.safe_load(example_file.read_text())

            # Directly use response as the assistant part
            assistant_part = example_data["response"].strip()

            # Create a simple user part, describing this as a scene description task
            user_part = "Please describe this scene based on the scene graph provided."

            return user_part, assistant_part
        else:
            # Use the default behavior of the parent class
            return super()._create_example_message(example_file)

    def _save_description_to_scene_dir(self, scene_name, description, prompt_tokens=0, completion_tokens=0):
        """
        Save the generated scene description configs/scenes/{scene_name}/descriptions/ Contents
        The format should be consistent with the parent class’s pre-generated descriptions
        """
        if not self.auto_save_descriptions:
            return

        # Create a description directory
        description_dir = Path(
            f"./configs/scenes/{scene_name}/descriptions/{self.provider}-{self.model.split('/')[-1]}-{self.temperature}"
        )
        if self.use_vlm:
            description_dir = description_dir.with_name(description_dir.name + "-vlm")
        description_dir.mkdir(parents=True, exist_ok=True)

        # Generate a file name with a timestamp and timeline suffix)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        description_file = description_dir / f"description-timeline-{timestamp}.txt"

        try:
            # Save the description content
            with open(description_file, "w", encoding="utf-8") as f:
                f.write(description)

            # Save the metadata
            metadata_file = description_dir / f"description-timeline-{timestamp}.yaml"
            metadata = {
                "description_file": description_file.name,
                "generated_at": datetime.now().isoformat(),
                "model": self.model,
                "provider": self.provider,
                "temperature": self.temperature,
                "description_type": self.description_type,
                "use_timeline_examples": self.use_timeline_examples,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"Saved scene description to {description_file}")
            self.logger.info(f"Saved metadata to {metadata_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save description to {description_file}: {e}")

    def _get_pregenerated_timeline_description(self, scene_name, idx=0):
        """
        Obtain the pre-generated timeline scene description
        Support the narrative_timeline type
        """
        description_dir = Path(
            f"./configs/scenes/{scene_name}/descriptions/{self.provider}-{self.model.split('/')[-1]}-{self.temperature}"
        )
        if self.use_vlm:
            description_dir = description_dir.with_name(description_dir.name + "-vlm")

        description_files = sorted(description_dir.glob("description-timeline-*.txt"))
        if not description_files:
            self.logger.warning(f"No pregenerated timeline scene description found in '{description_dir}'")
            return None

        description_file = description_files[idx]
        try:
            with open(description_file, "r", encoding="utf-8") as f:
                description = f.read()
            self.logger.info(f"Using pregenerated timeline scene description from '{description_file}'")
            return description
        except Exception as e:
            self.logger.warning(f"Failed to read pregenerated description from '{description_file}': {e}")
            return None

    def get_scene_description(self, scene_name, scene_graph, with_usage=False):
        """
        Overwrite the scene description acquisition method and add an automatic save function
        """
        # First, attempt to use the pre-generated description (supports narrative_timeline type)
        if self.cfg.scene_describer.use_pregenerated_description:
            description = self._get_pregenerated_timeline_description(scene_name)
            if description is not None:
                if with_usage:
                    return description, 0, 0
                return description

        # For non-narrative types, use the parent class methods
        if self.description_type not in ["narrative", "narrative_timeline"]:
            return super().get_scene_description(scene_name, scene_graph, with_usage)

        # Generate a new description
        self.logger.info(f"[SceneDescriberTimeline] Generating scene description for {scene_name}...")

        # Call the parent class method to generate the description
        if with_usage:
            description, prompt_tokens, completion_tokens = super().get_scene_description(
                scene_name, scene_graph, with_usage=True
            )
            # Save to the scene directory
            self._save_description_to_scene_dir(scene_name, description, prompt_tokens, completion_tokens)
            return description, prompt_tokens, completion_tokens
        else:
            description = super().get_scene_description(scene_name, scene_graph, with_usage=False)
            # Save to the scene directory
            self._save_description_to_scene_dir(scene_name, description)
            return description
