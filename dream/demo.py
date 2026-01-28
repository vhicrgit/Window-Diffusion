import torch
from transformers import AutoTokenizer
from model.modeling_dream import DreamModel
from transformers import __version__


model_path = 'Dream-org/Dream-v0-Instruct-7B'
model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager",).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


messages = [
    {"role": "user", "content": "Please write a Python class that implements a quick sort program."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

with torch.no_grad():
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        output_history=True,
        return_dict_in_generate=True,
        steps=256,
        temperature=0.0,
        top_p=None,
        alg="entropy",
        alg_temp=0.,
        o_win_size=128,
        i_win_size=32,
        refresh_cycle = 32,
        slide_window=True,
        early_stop=True
    )
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output)
]

print(generations[0].split(tokenizer.eos_token)[0])
