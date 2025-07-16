#!/usr/bin/env bash
# run_model_merger.sh

# 在这里定义所有要处理的目录
DIRS=(
  "/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/saves/lora_7b/global_step_9/actor"
  "/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/saves/lora_7b/global_step_18/actor"
  "/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/saves/lora_7b/global_step_27/actor"
  "/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/saves/lora_7b/global_step_36/actor"
  "/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/saves/lora_7b/global_step_45/actor"
  "/root/autodl-tmp/home/lizhuohang/reaserch/ICLR2026/EasyR1/saves/lora_7b/global_step_54/actor"

  # 如果后续还要加目录，继续在这里每行一个，注意最后一行也别漏了引号
)

echo "一共要处理 ${#DIRS[@]} 个目录"
echo "=============================="

for dir in "${DIRS[@]}"; do
  echo "正在处理: $dir"
  python scripts/model_merger.py --local_dir "$dir"
  if [ $? -eq 0 ]; then
    echo "  ✅ 完成: $dir"
  else
    echo "  ❌ 失败:  $dir"
  fi
  echo "------------------------------"
done

echo "全部处理结束"
