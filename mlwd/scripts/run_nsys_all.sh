#!/bin/bash
# MLWD nsys 批量采集脚本
# 每个实验点单独跑一次 nsys，避免分段问题
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
PHASES=(prefill decode)
NUM_RUNS=3

mkdir -p ${TRACE_DIR} ${OUTPUT_DIR}

# 清空旧结果
echo "{}" > ${RESULTS_JSON}

total=$((${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]} * ${#PHASES[@]}))
count=0

for b in "${BATCH_SIZES[@]}"; do
  for s in "${SEQ_LENGTHS[@]}"; do
    for phase in "${PHASES[@]}"; do
      count=$((count + 1))
      key="b${b}_s${s}_${phase}"
      trace_path="${TRACE_DIR}/${key}"
      sqlite_path="${TRACE_DIR}/${key}.sqlite"

      echo ""
      echo "============================================"
      echo "[${count}/${total}] ${key}"
      echo "============================================"

      # Step 1: nsys profile
      echo "[nsys] Profiling ${key}..."
      ${NSYS} profile \
        -o ${trace_path} --trace cuda,nvtx \
        --trace-fork-before-exec=true --cuda-graph-trace=node \
        --sample none --cpuctxsw none --force-overwrite true \
        python mlwd/run_profiling.py --model ${MODEL} \
          --batch_sizes ${b} --seq_lengths ${s} --phases ${phase} \
          --num_runs ${NUM_RUNS}

      # Step 2: 导出 SQLite
      echo "[nsys] Exporting SQLite..."
      ${NSYS} export \
        --type sqlite --output ${sqlite_path} \
        --force-overwrite true ${trace_path}.nsys-rep

      # Step 3: 解析
      echo "[nsys] Parsing..."
      PYTHONPATH=. python mlwd/parse_traces.py \
        --nsys ${sqlite_path} \
        --key ${key} \
        --output ${RESULTS_JSON}

      # 清理 trace 文件（保留 SQLite 和 JSON）
      rm -f ${trace_path}.nsys-rep ${trace_path}.qdstrm

      echo "[${count}/${total}] ${key} done."
    done
  done
done

echo ""
echo "============================================"
echo "All done! Results: ${RESULTS_JSON}"
echo "============================================"
cat ${RESULTS_JSON}
