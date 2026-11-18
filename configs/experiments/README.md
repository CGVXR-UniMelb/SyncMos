# Historical experiment presets

The single application sample is [`../sample.yaml`](../sample.yaml).
Start it with `python main.py`; use `python main.py --help` for modes.

These files preserve alternative parser/provider settings used by experiments and
factory tests. They are not additional startup examples. Their contents were moved
unchanged so existing experiment parameters remain reproducible.

| Previous path under `configs/` | Current path under this directory |
| --- | --- |
| `demo.yaml` | `standard.yaml` |
| `demo_batch_parser.yaml` | `batch_parser.yaml` |
| `demo_gptoss.yaml` | `gptoss.yaml` |
| `demo_our_parser.yaml` | `our_parser.yaml` |
| `demo_qwen.yaml` | `qwen.yaml` |
| `demo_qwen_thinking.yaml` | `qwen_thinking.yaml` |
| `demo_timeline.yaml` | `timeline.yaml` |

The `bbox` mode still defaults to `qwen.yaml`. Other modes default to `sample.yaml`,
the unchanged former `demo_timeline_batch.yaml`. Select a different preserved preset
with `--config configs/experiments/<name>.yaml` when reproducing an experiment.
Benchmark, scene and native sampling configurations keep their existing paths.
