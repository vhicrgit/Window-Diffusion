import torch
from transformers import AutoModel, AutoTokenizer

from model.modeling_llada import generate


def main():
    model_path = "GSAI-ML/LLaDA-8B-Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    except Exception:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    prompt_text = "Write a quicksort algorithm"
    prompt = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=True
    )["input_ids"].to(device)
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336

    out_ids = generate(
        model,
        prompt,
        steps=256,
        gen_length=256,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        refresh_cycle=32,
        window_tokens=64,
        active_tokens=16,
    )

    gen_suffix = tokenizer.decode(out_ids[0][prompt.shape[1] :], skip_special_tokens=False)
    gen_suffix_ids = tokenizer(gen_suffix, add_special_tokens=False)["input_ids"]
    gen_suffix_clean = tokenizer.decode(gen_suffix_ids, skip_special_tokens=True)

    print("=== Prompt ===")
    print(prompt_text)
    print("\n=== Generated ===")
    print(gen_suffix_clean)


if __name__ == "__main__":
    main()
