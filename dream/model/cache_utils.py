# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT and Qwen implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT and Qwen used by the Meta AI and Qwen team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Dream model."""

import math
from typing import List, Optional, Tuple, Union
import os
import torch
from transformers.cache_utils import Cache, DynamicCache

def select_by_sort(a: torch.Tensor,
                       rank: torch.Tensor,
                       L: torch.Tensor) -> torch.Tensor:

    perm = torch.argsort(rank)          
    a_sorted = a.index_select(2, perm)

    out = a_sorted.index_select(2, L) 
    return out


class KVCache(Cache):
    def __init__(self, num_hidden_layers: int, max_len: int, head_dim: int, device):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.max_len = max_len
        self.device = device
        self.dtype = torch.bfloat16
        
        self.k_read_buffer = None
        self.v_read_buffer = None

        self.key_cache = [
            torch.empty((1, 4, max_len, 128), device=device, dtype=self.dtype)
            for _ in range(self.num_hidden_layers)
        ]
        self.value_cache = [
            torch.empty((1, 4, max_len, 128), device=device, dtype=self.dtype)
            for _ in range(self.num_hidden_layers)
        ]
        

        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    
    def get_seq_length(self):
        max_len = 0
        for k in self.key_cache:
            if k is not None:
                max_len = max(max_len, k.shape[2])
        return max_len

    def refresh_cache(self):
        self.key_cache = [
            torch.empty((1, 4, self.max_len, 128), device=self.device, dtype=self.dtype)
            for _ in range(self.num_hidden_layers)
        ]
        self.value_cache = [
            torch.empty((1, 4, self.max_len, 128), device=self.device, dtype=self.dtype)
            for _ in range(self.num_hidden_layers)
        ]
        

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position", None)
        
        if cache_position is not None:
            key_cache = self.key_cache[layer_idx]     
            value_cache = self.value_cache[layer_idx] 

            pos = cache_position[0]

            index = pos.view(1, -1, 1).expand(
                key_states.shape[1],
                -1,                   
                key_states.shape[3], 
            )

            key_cache[0].scatter_(dim=1, index=index, src=key_states[0])
            value_cache[0].scatter_(dim=1, index=index, src=value_states[0])

    @torch.no_grad()
    def read(
        self,
        layer_idx: int,
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        cache_position: torch.LongTensor, 
    ):

        past_k = self.key_cache[layer_idx]
        past_v = self.value_cache[layer_idx]

        H, D = key_states.shape[1], key_states.shape[3]
        Kc = key_states.shape[2]
        Kp = cache_position.shape[1]
        total_len = Kc + Kp

        if (
            self.k_read_buffer is None
            or self.k_read_buffer.shape[2] < total_len
        ):
            self.k_read_buffer = key_states.new_empty(
                (1, H, total_len, D)
            )
            self.v_read_buffer = value_states.new_empty(
                (1, H, total_len, D)
            )

        k_buf = self.k_read_buffer
        v_buf = self.v_read_buffer

        k_buf[:, :, :Kc, :] = key_states
        v_buf[:, :, :Kc, :] = value_states

        pos = cache_position[0].long()

        cached_k = past_k.index_select(dim=2, index=pos)
        cached_v = past_v.index_select(dim=2, index=pos)

        k_buf[:, :, Kc:total_len, :] = cached_k
        v_buf[:, :, Kc:total_len, :] = cached_v

        return (
            k_buf[:, :, :total_len, :],
            v_buf[:, :, :total_len, :],
        )


    def update_and_read(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position", None)
        prv_cache_position = cache_kwargs.get("prv_cache_position", None)
        position_ids = cache_kwargs.get("position_ids", None)
        
        if cache_position is not None:
            key_cache = self.key_cache[layer_idx]    
            value_cache = self.value_cache[layer_idx]

            pos = cache_position[0] 

            index = pos.view(1, -1, 1).expand(
                key_states.shape[1], 
                -1,                   
                key_states.shape[3],
            )

            key_cache[0].scatter_(dim=1, index=index, src=key_states[0])
            value_cache[0].scatter_(dim=1, index=index, src=value_states[0])

        if prv_cache_position is not None:
            cur_len = position_ids.shape[1]
            return self.key_cache[layer_idx][:, :, :cur_len, :], self.value_cache[layer_idx][:, :, :cur_len, :]
        
        return key_states, value_states