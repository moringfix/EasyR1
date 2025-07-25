#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/root/autodl-tmp/home/Share/Model/LLM/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/data/base/Emotion_val.jsonl \
    data.val_files=/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/data/base/Emotion_test.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=22528 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=1 \
    trainer.logger=['console'] \
