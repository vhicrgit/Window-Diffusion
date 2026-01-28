import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Optional

def unwrap_model(model):
    """兼容 DDP / FSDP, 统一拿到真正的 HF 模型."""
    return model.module if hasattr(model, "module") else model

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int):
    B = mask_index.size(0)
    mask_num = mask_index.sum(dim=1, keepdim=True)     # (B,1)
    base = mask_num // steps
    rem  = mask_num % steps
    num_transfer_tokens = base.repeat(1, steps)
    for i in range(B):
        if rem[i] > 0:
            num_transfer_tokens[i, :int(rem[i])] += 1
    return num_transfer_tokens

@torch.no_grad()
def rope_apply_by_index(rope, pos_embed, x, idx):
    sin_full, cos_full = pos_embed  # [1,1,T_full,Dh]
    sin = sin_full[:, :, idx, :]  # [1,1,T,Dh]
    cos = cos_full[:, :, idx, :]
    # rope.apply_rotary_pos_emb 期望 sin/cos 广播到 [B,1,T,Dh]
    sin = sin.expand(x.size(0), -1, -1, -1)
    cos = cos.expand(x.size(0), -1, -1, -1)
    return rope.apply_rotary_pos_emb(sin, cos, x)



@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor
    
class KVCacheManager:

    def __init__(self):
        self.layer_caches: Dict[int, LayerKVCache] = {}

    def clear(self):
        self.layer_caches.clear()
            
    @torch.no_grad()
    def write(self, layer_id, k, v):
        if layer_id not in self.layer_caches:
            self.layer_caches[layer_id] = LayerKVCache(k, v)
        else:
            self.layer_caches[layer_id].key = k
            self.layer_caches[layer_id].value = v
            
@dataclass
class WindowPlan:
    active_idx: torch.Tensor
    window_idx: torch.Tensor  
    decoded_mask: torch.Tensor
    
def build_window_plan(
    x: torch.Tensor,
    mask_id: int,
    transfer_index: torch.Tensor,
    window_tokens: int = 64,
    active_tokens: int = 8,
):
    
    mask_index = (x == mask_id)              # [B,S]
    decoded_mask = ~mask_index               # [B,S]
    
    mask_rank = mask_index.long().cumsum(dim=1) - 1                    
    mask_rank = mask_rank.masked_fill(~mask_index, 10**9)              
    active_mask = (mask_rank >= 0) & (mask_rank < active_tokens) | transfer_index
    window_mask  = (mask_rank >= 0) & (mask_rank < window_tokens) | decoded_mask
    
    active_idx = active_mask[0].nonzero(as_tuple=True)[0]
    window_idx = window_mask[0].nonzero(as_tuple=True)[0]
    
    decoded_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    
    
    plan = WindowPlan(active_idx=active_idx, window_idx=window_idx, decoded_mask=decoded_mask)

    return plan


def update_window_plan(
    x: torch.Tensor,
    mask_id: int,
    plan: WindowPlan,
    active_tokens: int = 8,
):
    mask_index = (x == mask_id)              # [B,S]
    
    mask_rank = mask_index.long().cumsum(dim=1) - 1                    
    mask_rank = mask_rank.masked_fill(~mask_index, 10**9) 
                     
    active_mask = (mask_rank >= 0) & (mask_rank < active_tokens)  | plan.decoded_mask
    
    active_idx = active_mask[0].nonzero(as_tuple=True)[0]
    plan.active_idx = active_idx



@torch.no_grad()
def block_forward_full(
    block: nn.Module,
    x: torch.Tensor,
    block_id: int,
    cache_mgr: KVCacheManager
):
    x_normed = block.attn_norm(x)
    q = block.q_proj(x_normed)
    k = block.k_proj(x_normed)
    v = block.v_proj(x_normed)

    B, T, C = q.size()
    H   = block.config.n_heads
    KVH = block.config.effective_n_kv_heads

    q = q.view(B, T, H,   C // H).transpose(1, 2)   # [B,H,T,Dh]
    k = k.view(B, T, KVH, C // H).transpose(1, 2)   # [B,KVH,T,Dh]
    v = v.view(B, T, KVH, C // H).transpose(1, 2)
    

    if block.config.rope:
        q, k = block.rotary_emb(q, k)
        
    cache_mgr.write(block_id, k.contiguous(), v.contiguous())
        
    # GQA
    assert k.size(1) == v.size(1)
    if H != KVH:
        assert H % KVH == 0
        k = k.repeat_interleave(H // KVH, dim=1, output_size=H)
        v = v.repeat_interleave(H // KVH, dim=1, output_size=H)
    

    att = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )

    att = att.transpose(1, 2).contiguous().view(B, T, C)
    att_out = block.attn_out(att)

    x = x + block.dropout(att_out)
    x_normed = block.ff_norm(x)

    a = block.ff_proj(x_normed)
    up = block.up_proj(x_normed)
    a = block.act(a)
    a = a * up
    a = block.ff_out(a)
    a = block.dropout(a)
    x = x + a

    return x

@torch.no_grad()
def block_forward_prune(
    block: nn.Module,
    x: torch.Tensor,
    block_id: int,
    cache_mgr: KVCacheManager,
    plan: WindowPlan,
    pos_embed,
):
    B, _, C = x.size()
    H   = block.config.n_heads
    KVH = block.config.effective_n_kv_heads
    Dh = C // H
    
    x_normed = block.attn_norm(x)
    layer_cache = cache_mgr.layer_caches[block_id]
    rope = block.rotary_emb
    
    if plan.active_idx.numel() > 0:
        k = block.k_proj(x_normed).view(B, plan.active_idx.numel(), KVH, Dh).transpose(1, 2)  # [B,KVH,Ta,Dh]
        v = block.v_proj(x_normed).view(B, plan.active_idx.numel(), KVH, Dh).transpose(1, 2)
        k = rope_apply_by_index(rope, pos_embed, k, plan.active_idx.to(x_normed.device))
        
        
        layer_cache.key[:, :, plan.active_idx, :] = k
        layer_cache.value[:, :, plan.active_idx, :] = v
        
    k_window = layer_cache.key[:, :, plan.window_idx, :]     
    v_window = layer_cache.value[:, :, plan.window_idx, :]   
    
    q = block.q_proj(x_normed).view(B, plan.active_idx.numel(), H, Dh).transpose(1, 2)  # [B,H,Tg,Dh]
    q = rope_apply_by_index(rope, pos_embed, q, plan.active_idx.to(x.device))
        
    # GQA
    assert k_window.size(1) == v_window.size(1)
    if H != KVH:
        assert H % KVH == 0
        k_window = k_window.repeat_interleave(H // KVH, dim=1, output_size=H)
        v_window = v_window.repeat_interleave(H // KVH, dim=1, output_size=H)
        
    att = F.scaled_dot_product_attention(
        query=q, key=k_window, value=v_window,
        attn_mask=None, dropout_p=0.0, is_causal=False
    )
    
    att = att.transpose(1, 2).contiguous().view(B, plan.active_idx.numel(), C)
    att_out = block.attn_out(att)
    
    x = x + att_out
    
    x_norm2 = block.ff_norm(x)
    a = block.ff_proj(x_norm2)
    up = block.up_proj(x_norm2)
    a = block.act(a) * up
    a = block.ff_out(a)
    a = block.dropout(a)
    x = x + a
    
    return x


def llada_forward_full(
    tr: nn.Module,
    cfg,
    blocks,
    x: torch.Tensor,
    cache_mgr: KVCacheManager,
    
):  
    x = tr.wte(x)
    if cfg.input_emb_norm:
        x = x * (cfg.d_model**0.5)
    x = tr.emb_drop(x)

    for block_id, block in enumerate(blocks):
        x = block_forward_full(block, x, block_id, cache_mgr)
        
    x = tr.ln_f(x)
    
    if cfg.weight_tying:
        logits = F.linear(x, tr.wte.weight, None)  # type: ignore
    else:
        logits = tr.ff_out(x)  # type: ignore
    if cfg.scale_logits:
        logits.mul_(1 / math.sqrt(cfg.d_model))
        
    return logits

def llada_forward_prune(
    tr: nn.Module,
    cfg,
    blocks,
    pos_embed,
    x: torch.Tensor,
    cache_mgr: KVCacheManager,
    plan: WindowPlan,
):    
    x = tr.wte(x)
    if cfg.input_emb_norm:
        x = x * (cfg.d_model**0.5)
    x = tr.emb_drop(x)
    
    for block_id, block in enumerate(blocks):
        x = block_forward_prune(block, x, block_id, cache_mgr, plan, pos_embed)
        
    x = tr.ln_f(x)

    if cfg.weight_tying:
        logits = F.linear(x, tr.wte.weight, None)
    else:
        logits = tr.ff_out(x)
    if cfg.scale_logits:
        logits.mul_(1 / math.sqrt(cfg.d_model))
        

    return logits


@torch.no_grad()
def generate(
    model,
    prompt: torch.LongTensor,
    steps: int = 128,
    gen_length: int = 128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    window_size: int=5,
    window_tokens: int=64,
    active_tokens: int=8,
):
    
    device = model.device
    B = prompt.size(0)
    prompt_len = prompt.shape[1]
    S = prompt_len + gen_length
    T = S
    
    x = torch.full((1, S), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = prompt.clone()
    prompt_index = (x != mask_id)
    
    num_transfer_tokens = get_num_transfer_tokens((x == mask_id), steps)  # (B, steps)
    window_offset = 0
    
    cache_mgr = KVCacheManager()
    model = unwrap_model(model)
    tr = model.model.transformer
    cfg = model.config
    if hasattr(tr, "blocks"):
        blocks = list(tr.blocks)
    else:
        blocks = []
        for bg in tr.block_groups:
            blocks.extend(list(bg.blocks))
    pos_embed = blocks[0].rotary_emb.get_rotary_embedding(T, x.device)
    
    
    for i in range(steps):
        if window_offset == 0:
            window_offset += 1 
            mask_index = (x == mask_id)
            if cfg_scale > 0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
            else:
                x_ = x        
            
            logits = llada_forward_full(tr, cfg, blocks, x_, cache_mgr)
            
            if cfg_scale > 0:
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
                
            plan = build_window_plan(x, mask_id, transfer_index, window_tokens=window_tokens, active_tokens=active_tokens)
            
        else:
            window_offset += 1
            all_x = x.index_select(1, plan.active_idx)
            new_mask_index = (all_x == mask_id)  

            if cfg_scale > 0:
                un_x = all_x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([all_x, un_x], dim=0)
            else:
                x_ = all_x
                
            logits = llada_forward_prune(tr, cfg, blocks, pos_embed, x_, cache_mgr, plan)
            
            if cfg_scale > 0:
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
                
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p_g = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p_g = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
                
            confidence = torch.where(new_mask_index, x0_p_g, -np.inf)
                
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            all_x[transfer_index] = x0[transfer_index]

                
                
            b_idx, k_idx = torch.nonzero(transfer_index, as_tuple=True)  
            decoded_pos = plan.active_idx[k_idx]
            x[b_idx, decoded_pos] = all_x[b_idx, k_idx]
            plan.decoded_mask[b_idx, decoded_pos] = True
            
            update_window_plan(x, mask_id, plan, active_tokens)
            
            
        if window_offset == window_size:
            window_offset = 0
                
    return x

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    import torch
    from transformers import AutoModel, AutoTokenizer
    
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
        device_map="auto" 
    ).to(device).eval()
    
    prompt_text = "Write a quicksort algorithm"
    prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    
    gen_length = 256
    steps = gen_length  
    temperature = 0.0     
    cfg_scale = 0.0        
    remasking = "low_confidence" 

    active_tokens = 16
    window_tokens = 64
    window_size = 32
    
    out_ids = generate(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        window_size=window_size,
         window_tokens=window_tokens,
        active_tokens=active_tokens,
    )
    
    gen_suffix = tokenizer.decode(out_ids[0][prompt.shape[1]:], skip_special_tokens=False)
    gen_suffix_ids = tokenizer(gen_suffix, add_special_tokens=False)["input_ids"]
    gen_suffix_clean = tokenizer.decode(gen_suffix_ids, skip_special_tokens=True)
    
    print("=== Prompt ===")
    print(prompt_text)
    print("\n=== Generated ===")
    print(gen_suffix_clean)