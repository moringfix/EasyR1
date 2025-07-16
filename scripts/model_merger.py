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

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(local_path: str, remote_path: str):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    state_dict: Dict[str, List[torch.Tensor]] = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Cannot find key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: Tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: List[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    # with torch.device("meta"):
    #     model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    # assert isinstance(model, PreTrainedModel)
    # model.to_empty(device="cpu")

    # print(f"Saving model to {hf_path}...")
    # model.save_pretrained(hf_path, state_dict=state_dict)
    # del state_dict, model

    # === NEW: Detect whether this is a LoRA checkpoint =========================
    adapter_cfg_path = os.path.join(hf_path, "adapter_config.json")
    has_lora = os.path.exists(adapter_cfg_path) or any(
        k.endswith(("lora_A.weight", "lora_B.weight")) for k in state_dict
    )
    print(f"LoRA detected: {has_lora}")

    # ------- A) LoRA 分支 ------------------------------------------------------
    if has_lora:
        print(f"Detected LoRA checkpoint, merging LoRA into base model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        from peft import PeftConfig, get_peft_model

        # 1. 读取 LoRA 配置
        peft_cfg = PeftConfig.from_pretrained(hf_path)

        # 2. 用 meta device 构建 **空的** base 模型，节省内存
        with torch.device("meta"):
            base_model = AutoClass.from_config(config, torch_dtype=torch.bfloat16)
        base_model.to_empty(device="cpu")

        # 3. 给 base 模型插入 LoRA 结构
        lora_model = get_peft_model(base_model, peft_cfg)

        # 4. 加载合并后的 state_dict（含 base + LoRA）
        _missing, _unexpected = lora_model.load_state_dict(state_dict, strict=False)
        print(f"[merge] missing={len(_missing)}, unexpected={len(_unexpected)}")

        # 5. 合并并卸载 LoRA
        merged_model = lora_model.merge_and_unload()  # ← 关键一步

        # 6. 保存到一个新目录（避免覆盖原文件）
        step_id = local_dir.split("/")[-2].split("_")[-1]

        out_dir = os.path.join(local_dir, f"huggingface_merged_{step_id}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving merged model to {out_dir}")
        merged_model.save_pretrained(out_dir)         # 已含全部权重
        # tokenizer 等直接复用原目录
        from transformers import AutoTokenizer,AutoProcessor   
        tok = AutoTokenizer.from_pretrained(hf_path)
        tok.save_pretrained(out_dir)
        if os.path.exists(os.path.join(hf_path, "preprocessor_config.json")):
            proc = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
            proc.save_pretrained(out_dir)
    # ------- B) 非 LoRA 分支 ---------------------------------------------------
    else:
        with torch.device("meta"):
            model = AutoClass.from_config(config, torch_dtype=torch.bfloat16)
        model.to_empty(device="cpu")

        out_dir = os.path.join(local_dir, "huggingface_merged")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving model to {out_dir}")
        model.save_pretrained(out_dir, state_dict=state_dict)




    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
