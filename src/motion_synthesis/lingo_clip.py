"""Forward active text encoding to trusted LINGO code using the official CLIP library.

Cache code only, not models or embeddings. Native per-call loading/tokenization/
normalization/device/dtype behavior is unchanged; imports alone require no checkout.
"""

from functools import lru_cache
import os

from src.motion_synthesis.lingo_external import load_external_clip

_load_helpers = lru_cache(maxsize=1)(load_external_clip)


def get_clip_features(raw_text: str):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).get_clip_features(raw_text)
