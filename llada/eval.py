import logging
from datetime import timedelta
from typing import List, Optional, Type, TypeVar, Union

import torch
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

from model.modeling_llada import generate


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


@register_model("llada_window_diffusion")
class LLaDAWindowDiffusion(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_length: Optional[int] = 4096,
        gen_length: Optional[int] = 256,
        steps: Optional[int] = None,
        trust_remote_code: Optional[bool] = True,
        temperature: Optional[float] = 0.0,
        cfg_scale: Optional[float] = 0.0,
        remasking: Optional[str] = "low_confidence",
        mask_id: Optional[int] = None,
        window_tokens: Optional[int] = 64,
        active_tokens: Optional[int] = 16,
        refresh_cycle: Optional[int] = 32,
        attn_implementation: Optional[str] = "eager",
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))

        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError(
                "The current LLaDA Window-Diffusion implementation only supports batch_size=1."
            )
        self.max_length = int(max_length)
        self.gen_length = int(gen_length)
        self.steps = int(steps) if steps is not None else int(gen_length)
        self.temperature = float(temperature)
        self.cfg_scale = float(cfg_scale)
        self.remasking = remasking
        self.window_tokens = int(window_tokens)
        self.active_tokens = int(active_tokens)
        self.refresh_cycle = int(refresh_cycle)
        self.attn_implementation = attn_implementation
        self._mask_id_override = mask_id

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        device_list = set(
            ["cuda", "cpu"]
            + [f"cuda:{i}" for i in range(gpus)]
            + ["mps", "mps:0"]
            + [f"npu:{i}" for i in range(gpus)]
        )
        if accelerator.num_processes > 1:
            self._device = accelerator.device
        elif device in device_list:
            self._device = torch.device(device)
            if device in ("mps", "mps:0") and version.parse(
                torch.__version__
            ) < version.parse("2.1"):
                raise RuntimeError(
                    f"mps requires torch >= 2.1. You have {torch.__version__}"
                )
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._create_model_and_tokenizer(
            pretrained=pretrained,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

        if accelerator.num_processes > 1:
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )
        self.model = transformers.AutoModel.from_pretrained(pretrained, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self._mask_id_override is None:
            self.mask_id = (
                self.tokenizer.mask_token_id
                if self.tokenizer.mask_token_id is not None
                else 126336
            )
        else:
            self.mask_id = int(self._mask_id_override)

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

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _encode_context(self, context: str) -> torch.LongTensor:
        encoded = self.tokenizer(
            context,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max(self.max_length - self.gen_length, 1),
        )
        return encoded.input_ids.to(self.device)

    @torch.no_grad()
    def _generate_one(self, context: str) -> str:
        prompt_ids = self._encode_context(context)
        output_ids = generate(
            self.model,
            prompt_ids,
            steps=self.steps,
            gen_length=self.gen_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            remasking=self.remasking,
            mask_id=self.mask_id,
            window_tokens=self.window_tokens,
            active_tokens=self.active_tokens,
            refresh_cycle=self.refresh_cycle,
        )

        response = self.tokenizer.decode(
            output_ids[0][prompt_ids.shape[1] :].tolist(), skip_special_tokens=False
        )
        if self.tokenizer.eos_token:
            response = response.split(self.tokenizer.eos_token)[0]

        cleaned_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        return self.tokenizer.decode(cleaned_ids, skip_special_tokens=True)

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or self.rank != 0),
            desc="Running generate_until requests",
        )

        for req in requests:
            context, gen_args = req.arguments
            response = self._generate_one(context)

            until = gen_args.get("until", [])
            if isinstance(until, str):
                until = [until]
            for stop in until:
                response = response.split(stop)[0]

            res.append(response)
            pbar.update(1)
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "llada_window_diffusion only supports generate_until tasks in this artifact."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "llada_window_diffusion only supports generate_until tasks in this artifact."
        )


if __name__ == "__main__":
    cli_evaluate()
