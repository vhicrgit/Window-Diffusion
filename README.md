# Window-Diffusion: Efficient Inference for Diffusion Language Models

This repository contains the official reference implementation of **Window-Diffusion**, a training-free inference acceleration framework for masked discrete Diffusion Language Models (DLMs), proposed in our paper.

Window-Diffusion improves inference efficiency by **token-level selective computation** and **phase-level KV caching**, without modifying model architecture or retraining.

## Repository Structure

```
.
├── dream
│   ├── demo.py                 # Minimal Dream generation demo
│   ├── eval.py                 # lm-evaluation-harness adapter
│   ├── scripts
│   │   └── run_benchmarks.sh   # Dream benchmark entry point
│   └── model
│       ├── cache_utils.py
│       ├── configuration_dream.py
│       ├── generation_utils.py
│       └── modeling_dream.py
└── llada
    └── llada_window_diffusion.py
```

## Usage (Dream)

The Dream implementation exposes Window-Diffusion through
`model.diffusion_generate(...)`. The code is training-free: it loads a pretrained
Dream checkpoint and changes only the inference procedure.

### Quick Start

Install the basic runtime dependencies in your Python environment:

```
pip install torch transformers accelerate datasets evaluate lm-eval
```

Run the included Dream demo:

```
cd dream
python demo.py
```

By default, `demo.py` uses `Dream-org/Dream-v0-Instruct-7B`. If you use a local
checkpoint, edit `model_path` in `dream/demo.py`.

### Minimal Example

The following example is the same usage pattern as `dream/demo.py`:

```python
import torch
from transformers import AutoTokenizer
from model.modeling_dream import DreamModel

model_path = "Dream-org/Dream-v0-Instruct-7B"  # or a local checkpoint
device = "cuda"

model = DreamModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

messages = [{"role": "user", "content": "Please write a Python class that implements quick sort."}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True
)

input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

with torch.no_grad():
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        steps=256,
        temperature=0.0,
        alg="entropy",
        alg_temp=0.0,

        # Window-Diffusion knobs
        o_win_size=128,      # external window length
        i_win_size=32,       # internal window length (active tokens)
        refresh_cycle=32,    # phase-level KV refresh interval
        slide_window=True,
        early_stop=True,

        output_history=False,
        return_dict_in_generate=False,
    )

text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(text)
```

### Key knobs

- `o_win_size`: external window length; undecoded tokens outside this local prefix are pruned within the current phase.
- `i_win_size`: internal window length; these active tokens are updated and used for logits at the current step.
- `refresh_cycle`: phase-level KV-cache refresh interval.
- `slide_window`: whether the internal active-token window moves as decoding progresses.
- `early_stop`: enables adaptive-length generation once an EOS token is produced.

### Benchmark Evaluation

Dream benchmarks are run through `lm-evaluation-harness` with the local
`dream/eval.py` adapter, which registers the model name `dream_window_diffusion`.
The unified script supports GSM8K-CoT, MATH, HumanEval, and MBPP for Dream Base
and Dream Instruct.

Install the required evaluation dependencies in your Python environment:

```
pip install torch transformers accelerate datasets evaluate lm-eval
```

For code-generation tasks, allow the harness to execute task-provided tests:

```
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

Run one benchmark with the default HuggingFace checkpoint:

```
bash dream/scripts/run_benchmarks.sh --model-type base --task gsm8k
bash dream/scripts/run_benchmarks.sh --model-type instruct --task mbpp
```

Run all supported Dream benchmarks:

```
bash dream/scripts/run_benchmarks.sh --model-type base --task all
bash dream/scripts/run_benchmarks.sh --model-type instruct --task all
```

Use a local checkpoint:

```
MODEL_PATH=/path/to/Dream-v0-Base-7B \
  bash dream/scripts/run_benchmarks.sh --model-type base --task gsm8k
```

Run a quick smoke test with one example before launching a full benchmark:

```
CUDA_VISIBLE_DEVICES=0 LIMIT=1 \
MODEL_PATH=/path/to/Dream-v0-Base-7B \
  bash dream/scripts/run_benchmarks.sh --model-type base --task gsm8k
```

Without `LIMIT`, the script runs the full evaluation. Results are written to
`runs/dream/<model-type>/<task>` by default. Override `OUTPUT_ROOT`,
`CUDA_VISIBLE_DEVICES`, `DEVICE`, `MAIN_PROCESS_PORT`, or `LIMIT` as needed.

Supported task names:

| Script task | lm-eval task | Typical model type |
| --- | --- | --- |
| `gsm8k` | `gsm8k_cot` | base, instruct |
| `math` | `minerva_math` | base, instruct |
| `humaneval` | `humaneval` | base, instruct |
| `mbpp` | `mbpp` / `mbpp_instruct` | base / instruct |

## Usage (LLaDA)

### Quick Start

```
cd llada
python llada_window_diffusion.py
```

### Key knobs

The main generation function is:

- `window_tokens`: external window size (context tokens kept)
- `active_tokens`: internal window size (critical tokens updated per step)
