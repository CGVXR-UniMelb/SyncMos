from .scene_describer import SceneDescriber
from .scene_describer_timeline import SceneDescriberTimeline


def get_scene_describer(cfg):
    """
    Implement different SceneDescribers based on configuration
    """
    if cfg.scene_describer.description_type == "narrative_timeline":
        return SceneDescriberTimeline(cfg)
    return SceneDescriber(cfg)
