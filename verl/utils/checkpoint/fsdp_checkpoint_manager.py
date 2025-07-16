# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        state_dict_options = StateDictOptions(cpu_offload=True)
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
            options=state_dict_options,
        )
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def save_checkpoint(self, path: str, save_model_only: bool = False):
        path = self.local_mkdir(path)
        dist.barrier()

        # every rank will save its own model and optim shard
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        state_dict_options = StateDictOptions(cpu_offload=True)
        if save_model_only:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            torch.save(model_state_dict, model_path)
        else:
            model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
            extra_state_dict = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": self.get_rng_state(),
            }
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
            print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        dist.barrier()

        # if self.rank == 0:
        #     hf_path = os.path.join(path, "huggingface")
        #     os.makedirs(hf_path, exist_ok=True)
        #     assert isinstance(self.model._fsdp_wrapped_module, PreTrainedModel)
        #     self.model._fsdp_wrapped_module.config.save_pretrained(hf_path)
        #     self.model._fsdp_wrapped_module.generation_config.save_pretrained(hf_path)
        #     self.processing_class.save_pretrained(hf_path)


        # if self.rank == 0:
        #     # 最终将 huggingface checkpoint 写入到 path/huggingface
        #     hf_path = os.path.join(path, "huggingface")
        #     os.makedirs(hf_path, exist_ok=True)
        #     # 支持 PreTrainedModel 或 PeftModel（LoRA）
        #     from transformers import PreTrainedModel
        #     try:
        #         wrapped = self.model._fsdp_wrapped_module
        #         # 普通模型
        #         if isinstance(wrapped, PreTrainedModel):
        #             model_to_save = wrapped
        #         else:
        #             # 如果是 LoRA 包装
        #             from peft import PeftModel
        #             if isinstance(wrapped, PeftModel):
        #                 model_to_save = wrapped
        #             else:
        #                 raise AssertionError(f"不支持的模型类型：{type(wrapped)}，仅支持 PreTrainedModel 或 PeftModel")
        #     except ImportError:
        #         raise AssertionError("请确保 transformers 及 peft 已正确安装")

        #     # 保存：PreTrainedModel 会保存全量权重，PeftModel 会保存 adapter 配置和权重
        #     model_to_save.config.save_pretrained(hf_path)
        #     # 如果有 generation_config，则也保存
        #     if hasattr(model_to_save, "generation_config"):
        #         model_to_save.generation_config.save_pretrained(hf_path)
        #     # 保存 tokenizer/processor
        #     self.processing_class.save_pretrained(hf_path)

        if self.rank == 0:
            from peft import PeftModel
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)

            wrapped = self.model._fsdp_wrapped_module
            from transformers import PreTrainedModel
            if isinstance(wrapped, PreTrainedModel):
                model_to_save = wrapped
            elif isinstance(wrapped, PeftModel):
                model_to_save = wrapped
            else:
                raise AssertionError(f"Unsupported model type {type(wrapped)}")
            print(f"[rank-{self.rank}]: Saving HuggingFace model to {os.path.abspath(hf_path)}.")
            # （1）保存基础 config
            model_to_save.config.save_pretrained(hf_path)
            print(f"【debug】 11111111111111111111111111111111111111111111111111111")
            # （2）如果是 LoRA，务必写出 adapter_config.json
            if isinstance(model_to_save, PeftModel):
                # 下面两句二选一即可
                # ➤ 最简单：直接 save_pretrained 只写 LoRA adapter 文件（不会写 base 权重）
                print(f"【debug】 22222222222222222222222222222222222222222222222222")
                model_to_save.save_pretrained(hf_path, safe_serialization=False, state_dict=model_state_dict)
                #   - 这会在 huggingface/ 产生 adapter_config.json 及 adapter_model.bin
                #
                # ➤ 或者：显式保存 peft_config
                # model_to_save.peft_config.save_pretrained(hf_path)

            # （3）保存 generation_config（有就写）
            print(f"【debug】 33333333333333333333333333333333333333333333333333")
            if hasattr(model_to_save, "generation_config"):
                model_to_save.generation_config.save_pretrained(hf_path)
            print(f"【debug】 44444444444444444444444444444444444444444444444444")  
            # （4）保存 tokenizer / processor
            self.processing_class.save_pretrained(hf_path)

        dist.barrier()
