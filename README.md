# Window-Diffusion: Efficient Inference for Diffusion Language Models

This repository contains the official reference implementation of **Window-Diffusion**, a training-free inference acceleration framework for masked discrete Diffusion Language Models (DLMs), proposed in our paper.

Window-Diffusion improves inference efficiency by **token-level selective computation** and **phase-level KV caching**, without modifying model architecture or retraining.

## Repository Structure

```
.
├── dream
│   ├── demo.py
│   └── model
│       ├── cache_utils.py
│       ├── configuration_dream.py
│       ├── generation_utils.py
│       └── modeling_dream.py
└── llada
    └── llada_window_diffusion.py
```

## Usage (Dream)

### Quick Start

```
cd dream
python demo.py
```

### Minimal Example

```
Minimal generation example

The Dream implementation exposes Window-Diffusion via model.diffusion_generate(...) (see dream/demo.py):

import torch
from transformers import AutoTokenizer
from dream.model.modeling_dream import DreamModel

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
        refresh_cycle=32,    # cache refresh interval (if enabled in your build)
        slide_window=True,
        early_stop=True,

        output_history=False,
        return_dict_in_generate=False,
    )

text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(text)
```

### Key knobs

- `o_win_size`: external window length (range of undecoded context tokens kept around the decoding frontier)
- `i_win_size`: internal window length (number of active undecoded tokens updated per step)
- `refresh_cycle`: KV-cache refresh interval 
- `early_stop`: enable adaptive generation length, allowing the diffusion process to terminate adaptively  rather than using a fixed number of generation steps

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