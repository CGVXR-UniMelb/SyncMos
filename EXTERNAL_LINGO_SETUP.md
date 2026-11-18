# Running SyncMOS with External LINGO

This guide describes the current **inference/generation** setup through the single
`main.py` entrypoint. Run commands
from the root of **this Git repository**, not its parent workspace or the LINGO checkout.

SyncMOS loads native LINGO code from a trusted checkout while retaining its existing
sampling, DPS, time-warping, DTW, and cache controls. This is not a LINGO-free runtime.
Training and legacy standalone scripts are outside the scope of this guide.

## Before starting: what users must prepare

**Cloning this repository does not install a ready-to-run environment.** The repository
provides app code, runtime configurations, dependency specifications, and this guide.
Each user is responsible for installing compatible system/Python dependencies, obtaining
external resources under their providers' terms, and configuring local paths and credentials.
The original developer's conda environment, local symlinks, and caches do not transfer with Git.

| Requirement | What the user must provide |
| --- | --- |
| System environment | A compatible NVIDIA GPU/driver for generation, sufficient RAM/VRAM/storage, and required OS libraries. |
| Python environment | A dedicated environment and the dependencies locked in `pyproject.toml` / `uv.lock`; see section 1. |
| Installation tools | Git and the `uv` package manager; conda if following the example below. |
| External LINGO | A trusted checkout at the validated revision and an exported source path; see section 2. |
| Model/data resources | The three checkpoints, Dataset, licensed SMPL-X files, and HSSD meshes; see section 3. |
| Text encoding | The official CLIP dependency and its model cache, or permission/network access for its download. |
| LLM access | Credentials and billing/access for the configured provider when using LLM-backed generation. |
| Interactive viewer | A working graphical session and compatible OpenGL/display libraries; headless viewer setup is not covered here. |

The environment used for recorded model/kernel validation is Linux with Python 3.12 and
PyTorch 2.8.0 + CUDA 12.9. Dependency resolution on another platform does not establish
full generation support there. A CPU-only machine is not supported by the current
full-generation configuration. GPU memory requirements depend on the scene and workload;
this guide does not promise a minimum VRAM size or compatibility with every GPU/driver.

Follow sections 1–3 to prepare the environment, source, and resources, then run the file
preflight in section 4 before launching section 5. Obtain downloads from their official
providers; this repository does not provision or bundle those resources for other machines.

## 1. Environment

### Prepare a dedicated Python environment

The commands below are for a user setting up their own machine. Install
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and Git first if needed.
The project's package sources use Git to obtain the pinned official CLIP dependency.
Do not use a shared/base environment for the locked installation.

```bash
cd /path/to/my-Event-Driven-Storytelling

# Run this only if you do not already have the intended environment.
conda create -n syncmos python=3.12

conda activate syncmos
export ROOT_DIR="$PWD"
```

If the `syncmos` environment already exists, skip creation and activate it instead.
The environment name is a convention; a different name is fine if all subsequent commands
use the intended interpreter. Do not remove or recreate an existing environment just to
follow this example.

### Install the locked Python dependencies

With the dedicated conda environment active:

```bash
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
uv sync --locked --no-dev --python "$CONDA_PREFIX/bin/python"
uv pip check --python "$CONDA_PREFIX/bin/python"
```

The locked environment uses `pyproject.toml` and `uv.lock`, not the older requirements list.
`--locked` keeps the supplied lockfile unchanged. Installation may need network access to
package indexes and Git; a fully offline install requires separately prepared package caches.
The install provides Python dependencies, **not** LINGO checkpoints, motion data, SMPL-X,
HSSD, or a prepopulated CLIP model cache.

`uv sync` can replace packages and remove packages outside the lock in the target environment.
Review the target before running it; dependency synchronization is not necessary on every
launch. `UV_PROJECT_ENVIRONMENT` avoids silently installing into a separate project `.venv`
while you run the app with conda's Python. The examples below use `python` from the active
environment; do not accidentally use another interpreter from your system or editor.

Linux/Windows dependency sources select CUDA 12.9 PyTorch wheels. Install a compatible
NVIDIA driver independently using your platform/vendor's supported process. Python package
installation does not install or repair that driver. The local CUDA Toolkit version alone
does not determine whether the installed PyTorch CUDA runtime can use your GPU.

### Check the local CUDA runtime

```bash
python - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA generation is not ready in this environment."
PY
```

This diagnostic may initialize CUDA, but does not load model weights, download files, or
call an LLM API. A successful result is necessary for this configuration, not proof that
all models/scenes will fit in memory or that end-to-end generation works. This diagnostic
is separate from the non-GPU file-only preflight in section 4.

Setting only the top-level `device=cpu` is not a supported CPU-generation setup, because
the internal LINGO configuration and checkpoint loading retain CUDA defaults. OS-specific
missing-library or display errors must also be resolved locally; installing Python packages
alone does not guarantee that OpenCV, mesh backends, or the interactive viewer can run.
For dependency installation, see [Environment Setup](README.md#1-install-the-environment).

## 2. External source checkout

### Required checkout

Provide your own trusted `lingo-release` checkout. The revision used for compatibility
validation is:

```text
6de5ea649ca244e0f64bc23eacfe722e2b613034
```

Set the variable to the **checkout root containing `code/`**, not to `code/` itself:

```bash
export SYNCMOS_LINGO_CHECKOUT="/absolute/path/to/lingo-release"
git -C "$SYNCMOS_LINGO_CHECKOUT" rev-parse HEAD
```

The app does not automatically download LINGO or enforce this Git revision. Use the
validated source revision unless you intentionally evaluate another version. The loader
executes source from this checkout; import isolation is **not a security sandbox**.

The following files are needed by the active native-code boundaries:

```text
lingo-release/
└── code/
    ├── constants.py
    ├── transforms.py
    ├── utils.py
    ├── clip_utils.py
    ├── astar.py
    ├── sample_lingo.py
    ├── models/
    │   ├── synhsi.py
    │   └── joints_to_smplx.py
    └── datasets/
        ├── lingo.py
        ├── valid_idx_stand.npy
        └── valid_idx_sit.npy
```

`sample_lingo.py` is loaded for its geometry helper definitions, not run as an external
sampling process. Stand/sit indices are read from the external checkout.

### Optional local link

Instead of exporting the original location directly, you can create a convenient link:

```bash
mkdir -p external
if [ ! -e external/lingo-release ] && [ ! -L external/lingo-release ]; then
    ln -s "/absolute/path/to/lingo-release" external/lingo-release
fi
export SYNCMOS_LINGO_CHECKOUT="$PWD/external/lingo-release"
```

These commands do not replace an existing link. Inspect an existing destination before
using it. **Creating the link alone does not select external code: the export is required.**
Export it in each launch terminal and fully restart the app after code/binding changes.

## 3. Model weights and data

`SYNCMOS_LINGO_CHECKOUT` selects **source code only**. It does not change checkpoint,
dataset, SMPL-X, HSSD, or CLIP-cache locations. Obtain the resources separately from their
providers under the applicable terms. Do not commit or redistribute them with app code.

### Expected paths from the repository root

| Resource | Required local path |
| --- | --- |
| Motion diffusion / Unet checkpoint | `ckpts/model/bs128__ws16__scene_occ_two.pth` |
| Action-completion TimingModel checkpoint | `ckpts/scheduler/scheduler.pth` |
| JointsToSMPLX network checkpoint | `ckpts/joints2smplx/train_joint2smpl__input_84__hidden_128__all__last.pth` |
| Motion/language Dataset | `dataset/` (files below) |
| Male SMPL-X model for native fitting | `smpl_models/smplx/SMPLX_MALE.npz` |
| Male SMPL-X model for app rendering/body utilities | `data/SMPL/smplx/SMPLX_MALE.npz` |
| HSSD object meshes | `data/HSSD/objects/<first-character-of-model-id>/<model-id>.glb` |
| Official CLIP text encoder | `~/.cache/clip/ViT-L-14-336px.pt` (default cache) |

For the current 16-frame inference configuration, copy these six Dataset files,
keeping their contents and indices intact:

```text
dataset/
├── human_orient.npy
├── human_joints_aligned.npy
├── clip_features.npy
├── text2features_idx.pkl
├── norm_inter_and_loco__16frames.npy
└── language_motion_dict/
    └── language_motion_dict__inter_and_loco__16.pkl
```

The Dataset contains pickle metadata: use trusted resources only. Startup loads large
arrays into RAM, including approximately 1.9 GB of joint data. `clip_features.npy` supplies
archived Dataset embeddings; it does **not** replace the CLIP model needed for new prompts.
The inference adapter disables original LINGO `Scene`/`Scene_vis` loading and uses the app's
HSSD scene occupancy instead.

The full training collection is not required for this inference path. In particular,
AMASS/SAMP/Inter-X, LINGO `Scene`, `Scene_vis`, `Scene_mesh`, `human_pose.npy`, velocity
archives and training splits were not needed by the Windows smoke runs below. Do not
truncate the six retained arrays/metadata files independently: sampling initialization
uses the original segment/frame indices and the external stand/sit index lists.

### Link existing resources without moving them

Replace each placeholder with the directory containing the expected files. Do not run
these commands unchanged. Existing files, directories, and even broken symlinks are left
untouched; inspect/fix any existing incorrect destination separately.

```bash
if [ ! -e ckpts ] && [ ! -L ckpts ]; then
    ln -s "/absolute/path/to/ckpts" ckpts
fi
if [ ! -e dataset ] && [ ! -L dataset ]; then
    ln -s "/absolute/path/to/dataset" dataset
fi
if [ ! -e smpl_models ] && [ ! -L smpl_models ]; then
    ln -s "/absolute/path/to/smpl_models" smpl_models
fi
mkdir -p data/SMPL
if [ ! -e data/SMPL/smplx ] && [ ! -L data/SMPL/smplx ]; then
    ln -s "/absolute/path/to/smpl_models/smplx" data/SMPL/smplx
fi
```

Both SMPL-X locations can point to the same licensed files. They are separate lookup
paths: native fitting uses `smpl_models`, while app body-model utilities use `data/SMPL`.
Run from the repository root even when `ROOT_DIR` is exported; some paths are relative
to the working directory. The entrypoints use `ROOT_DIR` if already set, so explicitly
resetting it to `$PWD` avoids accidentally reusing another project's resource root.

### Example Linux workspace layout

An earlier Linux workspace used these links. This is a reference, not a requirement
that you reproduce the same parent-directory names or that they exist on a new machine:

```text
<workspace>/
├── lingo-release/                  # trusted native source checkout
├── lingo_utils/
│   ├── ckpts/                     # three checkpoints listed above
│   └── smpl_models/
│       └── smplx/                 # licensed SMPL-X model files
├── dataset/                       # LINGO Dataset resources
└── my-Event-Driven-Storytelling/   # run the app here
    ├── external/lingo-release -> ../../lingo-release
    ├── ckpts -> ../lingo_utils/ckpts
    ├── dataset -> ../dataset
    ├── smpl_models -> ../lingo_utils/smpl_models
    └── data/SMPL/smplx -> ../../../lingo_utils/smpl_models/smplx
```

### HSSD scene assets

For `scene_name=House`, provide the GLBs referenced by
`configs/scenes/House/scene_cfg.json`. The app also uses `scene.json`, `objects.json`, and
`grid_map.json` in that scene directory. Other scenes need their corresponding assets.
A precomputed occupancy cache or a separate `Scene_mesh/` folder is not a replacement for
the HSSD object meshes. Obtain assets through the [HSSD project](https://3dlg-hcvc.github.io/hssd/)
under its access/license requirements. No HSSD download is performed by this guide.

### CLIP and LLM credentials/network access

- Official OpenAI CLIP is a separate installed dependency. The native wrapper requests
  `ViT-L/14@336px`, using the default cache shown above. If its cached weight is missing or
  invalid, official CLIP can attempt a download. Arrange a valid cache before offline use;
  exporting the LINGO checkout does not disable downloads or move this cache.
- The default narrator/parser configuration uses OpenAI. Make the provider's API credentials
  available through the environment, e.g. `OPENAI_API_KEY`, using your normal secure workflow.
  Do not put credentials into commits or this document. ChatGPT subscription access is not
  API billing credit.
- `event_parser.use_vision=false` disables that parser's vision route. It does **not** disable
  text-LLM requests, guarantee zero cost/offline operation, or disable CLIP downloads.
- Some app modules call `python-dotenv` on import. This guide does not require creating or
  editing a `.env` file. Check your environment/configuration before allowing network calls.

## 4. File-only preflight (no models or network)

After the exports above, this checks source/resource file presence for `House`. It reads
scene JSON metadata only: no app import, `.env` read, checkpoint loading, pickle/NumPy
array loading, GPU initialization, download, or API call. It does not validate weight
contents, CUDA compatibility, or end-to-end generation.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path.cwd()
checkout_value = os.environ.get("SYNCMOS_LINGO_CHECKOUT")
if not checkout_value:
    raise SystemExit("Export SYNCMOS_LINGO_CHECKOUT before this check.")
checkout = Path(checkout_value).expanduser().resolve()
if Path(os.environ.get("ROOT_DIR", str(root))).resolve() != root:
    raise SystemExit("Run from the app repository root and export ROOT_DIR=\"$PWD\".")

sources = [
    "constants.py", "transforms.py", "utils.py", "clip_utils.py", "astar.py",
    "sample_lingo.py", "models/synhsi.py", "models/joints_to_smplx.py",
    "datasets/lingo.py", "datasets/valid_idx_stand.npy", "datasets/valid_idx_sit.npy",
]
resources = [
    "main.py",
    "ckpts/model/bs128__ws16__scene_occ_two.pth",
    "ckpts/scheduler/scheduler.pth",
    "ckpts/joints2smplx/train_joint2smpl__input_84__hidden_128__all__last.pth",
    "dataset/human_orient.npy", "dataset/human_joints_aligned.npy",
    "dataset/clip_features.npy", "dataset/text2features_idx.pkl",
    "dataset/norm_inter_and_loco__16frames.npy",
    "dataset/language_motion_dict/language_motion_dict__inter_and_loco__16.pkl",
    "smpl_models/smplx/SMPLX_MALE.npz", "data/SMPL/smplx/SMPLX_MALE.npz",
    "configs/scenes/House/scene_cfg.json", "configs/scenes/House/scene.json",
    "configs/scenes/House/objects.json", "configs/scenes/House/grid_map.json",
]
required = [checkout / "code" / name for name in sources]
required += [root / name for name in resources]
scene_path = root / "configs/scenes/House/scene_cfg.json"
if scene_path.is_file():
    scene = json.loads(scene_path.read_text())
    for obj in scene.values():
        model_id = obj["model_id"]
        required.append(root / "data/HSSD/objects" / model_id[0] / f"{model_id}.glb")
missing = sorted({str(path) for path in required if not path.is_file()})
clip_weight = Path.home() / ".cache/clip/ViT-L-14-336px.pt"
print("CLIP cached weight present:", clip_weight.is_file())
if missing:
    print("Missing required files:\n" + "\n".join(missing))
    raise SystemExit(1)
print("Source, Dataset, checkpoint, SMPL-X, and House mesh files are present.")
print("File presence only; CLIP integrity and model/app execution are not checked.")
PY
```

The CLIP line is informational: `False` means new-prompt encoding may attempt a download.
A `True` result checks presence only; official CLIP still validates its cached download.

## 5. Launch the app

### Interactive generation/viewer

From the app repository root, with resources already linked:

```bash
conda activate syncmos
export ROOT_DIR="$PWD"
export SYNCMOS_LINGO_CHECKOUT="$PWD/external/lingo-release"

python -m main \
  scene_name=House num_characters=2 event_parser.use_vision=false
```

If you did not create the optional link, use the real checkout path in the export instead.
`python -m main` opens the unified viewer using `configs/sample.yaml`. The viewer
does not automatically start generation: edit **User Instruction**, then click
**Generate New Motion**. **Continue From Loaded** resumes from a loaded motion source.
The launch example overrides the default four-character configuration with two characters.

### Offline generation

```bash
python -m main --mode generate \
  --config ./configs/sample.yaml \
  --motion-dir motion \
  --max-frames 300 \
  scene_name=House num_characters=2 event_parser.use_vision=false
```

This generates and saves motion without the interactive viewer, but it still needs the
same external code, GPU/resources, and applicable provider credentials. There is no
`--text` option. The interactive instruction editor is the documented way to provide a
custom instruction. `--max-frames` bounds generated output; it is not an API-cost cap.

### Where selection is configured

Active factory targets and resource defaults live in
`src/motion_synthesis/motion_generator/configs/lingo/`:

- `config_sample_lingo.yaml`: inference mode, device, Unet and TimingModel checkpoint paths.
- `model/model_smplx.yaml`: JointsToSMPLX checkpoint path.
- `model/*.yaml`, `dataset/lingo.yaml`, `sampler/pelvis.yaml`: local external-code factories
  with `checkout: ${oc.env:SYNCMOS_LINGO_CHECKOUT,null}`.
- `dataset/lingo.yaml`: Dataset directory under `${oc.env:ROOT_DIR}/dataset`.

The application entrypoint calls the existing pipeline; it does not need a separate external-LINGO
launch flag. Keep these runtime YAMLs even if old standalone/training code is removed.
Use the file-only preflight above to check resource placement before launching the app.

## 6. Troubleshooting and scope

| Symptom | Check |
| --- | --- |
| `Set SYNCMOS_LINGO_CHECKOUT ...` | Export the variable in the same terminal/process that launches the app. |
| `Missing external LINGO source` | Select the checkout root, not `code/`; check the files and revision above. |
| Missing checkpoint/Dataset/SMPL-X file | Check the app-root links and `ROOT_DIR`; checkout selection does not relocate these resources. |
| Missing HSSD `.glb` | Supply every mesh referenced by the selected scene; a mesh/occupancy cache is insufficient. |
| CUDA/default `cuda:0` error | Use the configured CUDA environment and a compatible driver; a top-level CPU override is insufficient. |
| CLIP download starts | Populate the official model cache before offline use; Dataset embeddings are not CLIP weights. |
| Provider authentication/billing error | Check your selected LLM provider and credentials; disabling vision does not disable text requests. |
| Changes seem unapplied | Fully stop/restart the app; already imported Python classes are not updated by a Git commit. |

A Git commit records selected files; running Python uses the files in the current working
tree. A separate clone only receives committed code, not your resource links, exported
environment variables, weights, datasets, or uncommitted changes.

Motion generation uses the local sampling/time-warp controls and external LINGO adapters.
Inspect action timing, object ownership and playback quality when preparing a new scene.
External source and resource licenses remain separate from this repository's own code license.
