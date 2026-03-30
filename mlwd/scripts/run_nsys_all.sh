#!/bin/bash
# MLWD nsys 批量采集脚本
# 每个 (batch_size, seq_len) 组合跑一次 nsys
# 一次推理同时包含 prefill + decode，不需要分开
#
# Usage:
#   bash mlwd/scripts/run_nsys_all.sh

set -e

NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
MODEL=/data/Qwen/Qwen2.5-7B-Instruct
OUTPUT_DIR=mlwd_output
TRACE_DIR=/tmp/mlwd_nsys_traces
RESULTS_JSON=${OUTPUT_DIR}/mlwd_results_nsys.json

BATCH_SIZES=(1 4)
SEQ_LENGTHS=(32 64 128)
NUM_RUNS=3
# generate 的 max_tokens，控制 decode 长度
MAX_TOKENS=32

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
    trace_path="${TRACE_DIR}/${key}"
    sqlite_path="${TRACE_DIR}/${key}.sqlite"

    # 跳过已采集的
    if python3 -c "import json; d=json.load(open('${RESULTS_JSON}')); exit(0 if '${key}' in d and d['${key}'].get('num_kernels',0)>10 else 1)" 2>/dev/null; then
      echo "[${count}/${total}] ${key} - SKIP"
      continue
    fi

    echo ""
    echo "============================================"
    echo "[${count}/${total}] ${key} (prefill s=${s} + decode ${MAX_TOKENS} tokens)"
    echo "============================================"

    # nsys profile: prefill 用长 prompt，然后 decode 生成 MAX_TOKENS 个 token
    ${NSYS} profile \
      -o ${trace_path} --trace cuda,nvtx \
      --trace-fork-before-exec=true --cuda-graph-trace=node \
      --sample none --cpuctxsw none --force-overwrite true \
      python mlwd/run_profiling.py --model ${MODEL} \
        --batch_sizes ${b} --seq_lengths ${s} --phases prefill \
        --num_runs ${NUM_RUNS} --max_tokens ${MAX_TOKENS} 2>&1 | tail -5

    # 导出
    ${NSYS} export \
      --type sqlite --output ${sqlite_path} \
      --force-overwrite true ${trace_path}.nsys-rep 2>&1 | tail -2

    # 解析
    PYTHONPATH=. python mlwd/parse_traces.py \
      --nsys ${sqlite_path} \
      --key ${key} \
      --output ${RESULTS_JSON}

    rm -f ${trace_path}.nsys-rep ${trace_path}.qdstrm

    echo "[${count}/${total}] ${key} done."
  done
done

echo ""
echo "============================================"
echo "All done! Results: ${RESULTS_JSON}"
echo "Total entries: $(python3 -c "import json; print(len(json.load(open('${RESULTS_JSON}'))))")"
echo "============================================"
