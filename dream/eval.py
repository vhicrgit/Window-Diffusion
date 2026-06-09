import logging
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from lm_eval import utils
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from packaging import version
from tqdm import tqdm

from model.modeling_dream import DreamModel


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


@register_model("dream_window_diffusion")
class DreamWindowDiffusion(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = True,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        o_win_size: Optional[int] = 128,
        i_win_size: Optional[int] = 16,
        refresh_cycle: Optional[int] = 32,
        early_stop: Optional[bool] = False,
        slide_window: Optional[bool] = True,
        apply_chat_template: Optional[Union[bool, str]] = False,
        attn_implementation: Optional[str] = "eager",
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        self.batch_size_per_gpu = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.add_bos_token = add_bos_token
        self.diffusion_steps = int(diffusion_steps)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.o_win_size = int(o_win_size)
        self.i_win_size = int(i_win_size)
        self.refresh_cycle = int(refresh_cycle)
        self.early_stop = early_stop
        self.slide_window = slide_window
        self.if_apply_chat_template = bool(apply_chat_template)

        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = int(mc_num)
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        if not (parallelize or accelerator.num_processes > 1):
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            if device != "cuda":
                eval_logger.info(
                    f"Using accelerate; explicit device '{device}' will be overridden."
                )
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self._create_model_and_tokenizer(
            pretrained, dtype, trust_remote_code, attn_implementation
        )

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    self.model.to(self.device)
            if gpus > 1 and accelerator.num_processes > 1:
                self._device = torch.device(f"{accelerator.device}")
                self.accelerator = accelerator
                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes
            else:
                self._rank = 0
                self._world_size = 1
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(
        self, pretrained, dtype, trust_remote_code, attn_implementation
    ):
        model_kwargs = {
            "torch_dtype": get_dtype(dtype),
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = DreamModel.from_pretrained(pretrained, **model_kwargs).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args.update({k: v for k, v in additional_config.items() if v is not None})
        return cls(**args)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _format_prompts(self, prompts: List[str]) -> List[str]:
        if self.if_apply_chat_template:
            return prompts
        if self.add_bos_token and self.tokenizer.bos_token is not None:
            return [self.tokenizer.bos_token + prompt for prompt in prompts]
        return prompts

    def _tokenize_prompts(self, prompts: List[str]):
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=len(prompts) > 1,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side

        prompt_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        max_prompt_len = self.max_length - self.max_new_tokens
        if prompt_ids.shape[-1] > max_prompt_len:
            eval_logger.warning(
                "Prompt length %s is larger than %s; truncating on the left.",
                prompt_ids.shape[-1],
                max_prompt_len,
            )
            prompt_ids = prompt_ids[:, -max_prompt_len:]
            attention_mask = attention_mask[:, -max_prompt_len:]
        return prompt_ids.to(self.device), attention_mask.to(self.device)

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        prompts = self._format_prompts(prompts)
        prompt_ids, attention_mask = self._tokenize_prompts(prompts)

        output_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=False,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            o_win_size=self.o_win_size,
            i_win_size=self.i_win_size,
            refresh_cycle=self.refresh_cycle,
            early_stop=self.early_stop,
            slide_window=self.slide_window,
        )

        responses = []
        prompt_len = prompt_ids.shape[1]
        for generated in output_ids:
            response = self.tokenizer.decode(generated[prompt_len:].tolist())
            if self.tokenizer.eos_token:
                response = response.split(self.tokenizer.eos_token)[0]
            responses.append(response)
        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or self.rank != 0),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(list(contexts))

            for i, response in enumerate(responses):
                if not self.escape_until:
                    until = gen_args[i].get("until", [])
                    if isinstance(until, str):
                        until = [until]
                    for stop in until:
                        response = response.split(stop)[0]
                responses[i] = response

            res.extend(responses)
            pbar.update(len(contexts))
        return res

    def _forward_process(self, batch):
        bsz, seq_len = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(bsz, device=batch.device).float()
        t = (u0 + indices / bsz) % 1
        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, seq_len)

        mask_indices = torch.rand((bsz, seq_len), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.classifier_free_guidance > 1.0:
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.classifier_free_guidance * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    def _encode_pair(self, context, continuation):
        if self.add_bos_token and self.tokenizer.bos_token is not None:
            context = self.tokenizer.bos_token + context

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)
        continuation_enc = whole_enc[len(context_enc) :]

        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            context_remain = len(context_enc) - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                context_enc = []
                continuation_enc = whole_enc[-self.max_length :]
        return context_enc, continuation_enc

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if len(prefix) == 0:
            seq = target[None, :]
        else:
            seq = torch.cat([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        if self.log_type == "btf":
            prompt_index = ~prompt_index

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == "ftb":
                perturbed_seq[:, -len(target) :] = perturbed_seq_[:, -len(target) :]
            elif self.log_type == "btf":
                perturbed_seq[:, : len(prefix)] = perturbed_seq_[:, : len(prefix)]
            elif self.log_type == "union":
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = (
                F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none")
                / p_mask[mask_indices]
            )
            loss_acc.append((loss.sum() / self.batch_size).item())
        return sum(loss_acc) / len(loss_acc)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        out = []
        for req in tqdm(requests, desc="Computing likelihood..."):
            prefix, target = self._encode_pair(req.args[0], req.args[1])
            prefix = torch.tensor(prefix, dtype=torch.long)
            target = torch.tensor(target, dtype=torch.long)
            if self.nll_type != "mc":
                raise NotImplementedError(self.nll_type)
            ll = -self._eval_target_nll_mc(prefix, target)
            if self.log_type == "union":
                ll = ll / (len(target) + len(prefix))
            out.append((ll, False))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()
