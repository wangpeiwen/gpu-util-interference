#!/bin/bash
# MLWD ncu 批量采集脚本
# 每个 (batch_size, seq_len) 组合跑一次 ncu
# 采集 CI, L2 hit rate, IPC
#
# 注意：ncu 使用 application replay，每个 metric 重跑一次完整推理
# 7B 模型下每个实验点预计 30-60 分钟
#
# Usage:
#   bash mlwd/scripts/run_ncu_all.sh
#   nohup bash mlwd/scripts/run_ncu_all.sh > mlwd_output/ncu_log.txt 2>&1 &

set -e

NCU=$(which ncu 2>/dev/null || echo "/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu")
MODEL=/data/Qwen/Qwen2.5-7B-Instruct
OUTPUT_DIR=mlwd_output
TRACE_DIR=/tmp/mlwd_ncu_traces
RESULTS_JSON=${OUTPUT_DIR}/mlwd_results_ncu.json

BATCH_SIZES=(1 4)
SEQ_LENGTHS=(32 64 128)

# ncu metrics (V100 SM70)
METRICS="sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
METRICS="${METRICS},sm__sass_thread_inst_executed_op_hfma_pred_on.sum"
METRICS="${METRICS},dram__bytes.sum"
METRICS="${METRICS},lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
METRICS="${METRICS},lts__t_sectors_srcunit_tex_op_read.sum"
METRICS="${METRICS},sm__inst_executed.avg.per_cycle_active"
METRICS="${METRICS},gpu__time_duration.sum"

# application replay + 跳过模型加载 kernel + 只采集少量推理 kernel
LAUNCH_SKIP=500
LAUNCH_COUNT=50

mkdir -p ${TRACE_DIR} ${OUTPUT_DIR}

if [ ! -f ${RESULTS_JSON} ]; then
  echo "{}" > ${RESULTS_JSON}
fi

total=$((${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]}))
count=0

for b in "${BATCH_SIZES[@]}"; do
  for s in "${SEQ_LENGTHS[@]}"; do
    count=$((count + 1))
    key="b${b}_s${s}"
    csv_path="${TRACE_DIR}/${key}.csv"

    # 跳过已采集的
    if python3 -c "import json; d=json.load(open('${RESULTS_JSON}')); exit(0 if '${key}' in d and d['${key}'].get('num_profiled_kernels',0)>0 else 1)" 2>/dev/null; then
      echo "[${count}/${total}] ${key} - SKIP"
      continue
    fi

    echo ""
    echo "============================================"
    echo "[${count}/${total}] ${key} (ncu profiling)"
    echo "============================================"

    # ncu profile (VLLM_USE_V1=0 强制单进程引擎)
    VLLM_USE_V1=0 ${NCU} \
      --csv \
      --metrics ${METRICS} \
      --kernel-name-base demangled \
      --replay-mode application \
      --launch-skip ${LAUNCH_SKIP} \
      --launch-count ${LAUNCH_COUNT} \
      --target-processes all \
      --log-file ${csv_path} \
      python mlwd/run_profiling.py --model ${MODEL} \
        --batch_sizes ${b} --seq_lengths ${s} --phases prefill \
        --num_runs 1 --warmup_runs 1 --max_tokens 32 \
      2>&1 | tail -10

    # 检查 CSV 是否生成
    if [ ! -f ${csv_path} ] || [ ! -s ${csv_path} ]; then
      echo "[${count}/${total}] ${key} - FAILED (no CSV output)"
      continue
    fi

    # 解析
    PYTHONPATH=. python mlwd/parse_traces.py \
      --ncu ${csv_path} \
      --prefix ${key} \
      --output ${RESULTS_JSON}

    echo "[${count}/${total}] ${key} done."
  done
done

echo ""
echo "============================================"
echo "All done! Results: ${RESULTS_JSON}"
echo "Total entries: $(python3 -c "import json; print(len(json.load(open('${RESULTS_JSON}'))))")"
echo "============================================"
