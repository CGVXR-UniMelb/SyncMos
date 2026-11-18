"""Narrow in-process loading of a trusted, user-supplied LINGO checkout.

Only the model closures, standalone Dataset/CLIP wrapper, constants/transforms/utils
math closure and the fixed sample_lingo definition/import closure are supported.
Import isolation is NOT a security sandbox. A dedicated fitting loader binds only
its isolated module's SMPL directory and quiet loss-log policy; native function bodies
remain unchanged. Original scene construction, direct Hydra imports of external classes,
full-module pickling and general package imports are unsupported. Hydra may call the
constructor-only local factories.
"""

import builtins
import importlib.util
from pathlib import Path


def _load_external_model(checkout, model_module):
    """Execute a fixed closure without global aliases, sys.path or chdir."""
    code = Path(checkout).expanduser().resolve() / "code"
    names = (
        (model_module,)
        if model_module in ("datasets.lingo", "clip_utils")
        else ("constants", "transforms") + (("utils",) if model_module == "models.synhsi" else ()) + (model_module,)
    )
    if model_module == "sample_lingo":
        names = (
            "constants",
            "transforms",
            "utils",
            "models.synhsi",
            "models.joints_to_smplx",
            "clip_utils",
            "datasets.lingo",
            "astar",
            "sample_lingo",
        )
    files = {name: code / f"{name.replace('.', '/')}.py" for name in names}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing external LINGO source: {path}")
    modules = {}

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "cv2" and model_module == "sample_lingo":
            # Unused astar dependency: OpenCV import mutates Qt/LD environment globally.
            # Resolve the real library only on attribute access, never during get_mat.
            from types import ModuleType

            module = ModuleType("cv2")
            module.__getattr__ = lambda attr: getattr(builtins.__import__("cv2"), attr)
            return module
        if level == 0 and name in files:
            return load(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(name):
        if name not in modules:
            spec = importlib.util.spec_from_file_location(f"_syncmos_external_lingo.{name}", files[name])
            module = importlib.util.module_from_spec(spec)
            module.__dict__["__builtins__"] = {**vars(builtins), "__import__": local_import}
            modules[name] = module
            spec.loader.exec_module(module)
        return modules[name]

    return load(model_module)


def load_external_synhsi(checkout):
    """Load the unchanged native Unet/TimingModel/sampler module."""
    return _load_external_model(checkout, "models.synhsi")


def load_external_dataset(checkout):
    """Load native Dataset methods; no constructor or resource access."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return _load_external_model(checkout, "datasets.lingo")


def create_external_dataset(checkout=None, **dataset_options):
    """Use native Dataset loading/methods with the explicit inference HSSD bridge."""
    upstream = load_external_dataset(checkout)
    from src.motion_synthesis.lingo_dataset import DatasetSceneAdapter

    class ExternalDataset(DatasetSceneAdapter, upstream.LingoDataset):
        pass

    return ExternalDataset(valid_indices_dir=Path(checkout).expanduser().resolve() / "code/datasets", **dataset_options)


def create_external_joints_to_smplx(checkout=None, **model_options):
    """Construct only the native network; keep local fitting and init_model setup."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return _load_external_model(checkout, "models.joints_to_smplx").JointsToSMPLX(**model_options)


def create_external_unet(checkout=None, **model_options):
    """Construct once; the existing init_model owns checkpoint/device/eval setup."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return load_external_synhsi(checkout).Unet(**model_options)


def create_external_timing_model(checkout=None, **model_options):
    """Construct once; callers retain checkpoint/device/eval and RNG ordering."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return load_external_synhsi(checkout).TimingModel(**model_options)


def create_external_sampler(checkout=None, **sampler_options):
    """Attach SyncMOS extensions to an unchanged native external sampler base."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    upstream = load_external_synhsi(checkout)
    from src.motion_synthesis.lingo_sampling import SamplingExtensions

    class ExternalSampler(SamplingExtensions, upstream.Sampler):
        _extract = staticmethod(upstream.extract)
        _native_sample_loop = staticmethod(upstream.Sampler.p_sample_loop)

    return ExternalSampler(**sampler_options)


def load_external_unet(checkout, checkpoint, device, **model_options):
    """Return the native external model, strict-loaded exactly as the original."""
    import torch

    model = create_external_unet(checkout, **model_options)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_external_joint_fitting(checkout):
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    module = _load_external_model(checkout, "models.joints_to_smplx")
    module.SMPL_DIR = "smpl_models"
    module.__dict__["__builtins__"]["print"] = lambda *args, **kwargs: None
    return module


def load_external_clip(checkout):
    """Load native text helpers only; never load CLIP weights or encode at import."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return _load_external_model(checkout, "clip_utils")


def load_external_geometry(checkout):
    """Load the unchanged script as a private module; never call its sample entrypoint."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return _load_external_model(checkout, "sample_lingo")


def load_external_math(checkout):
    """Load native math only; preserve resource globals, diagnostics and autograd."""
    if not checkout:
        raise ValueError("Set SYNCMOS_LINGO_CHECKOUT to your trusted lingo-release checkout.")
    return _load_external_model(checkout, "utils")
