# MLWD nsys 采集流程

## 前置条件

- 模型路径：`/data/Qwen/Qwen2.5-7B-Instruct`
- nsys 路径：`/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`
- 工作目录：`gpu-util-interference/`
- 已安装 vLLM、transformers

## Step 1: 验证推理正常

```bash
PYTHONPATH=. python mlwd/run_profiling.py \
  --model /data/Qwen/Qwen2.5-7B-Instruct \
  --batch_sizes 1 --seq_lengths 32 --num_runs 1
```

确认输出 `Profiling complete.` 且无报错。

## Step 2: nsys 采集

关键参数：
- `--trace-fork-before-exec=true`：跟踪 vLLM EngineCore 子进程的 GPU 操作
- `--trace cuda,nvtx`：捕获 CUDA kernel 和 NVTX marker
- `--force-overwrite true`：覆盖已有 trace 文件

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  -o /tmp/mlwd_trace --trace cuda,nvtx \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --sample none --cpuctxsw none --force-overwrite true \
  python mlwd/run_profiling.py --model /data/Qwen/Qwen2.5-7B-Instruct \
    --batch_sizes 1 4 --seq_lengths 32 64 128 \
    --output_meta /tmp/mlwd_meta.json
```

根据需要调整 `--batch_sizes` 和 `--seq_lengths`。注意 V100 32GB 下 decode 阶段 seq_len 不宜超过 256（OOM 风险）。

输出文件：`/tmp/mlwd_trace.nsys-rep`

## Step 3: 查看 kernel 统计（可选）

快速确认是否捕获到 kernel 数据：

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys stats \
  --report cuda_gpu_kern_sum --force-export=true \
  /tmp/mlwd_trace.nsys-rep
```

应能看到 `cutlass.*gemm`、`kernel_unified_attention` 等 kernel 的统计表。

## Step 4: 导出 SQLite

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys export \
  --type sqlite --output /tmp/mlwd_trace.sqlite \
  --force-overwrite true /tmp/mlwd_trace.nsys-rep
```

## Step 5: 解析提取 MLWD 特征

```bash
PYTHONPATH=. python mlwd/parse_traces.py \
  --nsys /tmp/mlwd_trace.sqlite \
  --output mlwd_output/mlwd_results.json
```

提取的特征（per 实验点）：
- `t_attn`：Attention kernel 平均时延 (μs)
- `t_ffn`：FFN kernel 平均时延 (μs)
- `g_launch`：平均 kernel launch 间隔 (μs)
- `r_attn`：Attention 时间占比
- `r_ffn`：FFN 时间占比
- `f_switch`：计算-访存交替频率 (次/秒)

## Step 6: 查看结果

```bash
cat mlwd_output/mlwd_results.json
```

或用 `python mlwd/inspect_nsys_db.py /tmp/mlwd_trace.sqlite` 查看原始数据库内容。

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `does not contain CUDA kernel data` | 缺少 `--trace-fork-before-exec=true` | 重新跑 Step 2 |
| `num_attn_kernels: 0` | kernel 分类器未匹配 | 用 Step 3 查看实际 kernel 名，更新 `kernel_classifier.py` |
| `file is not a database` | SQLite 导出失败或文件损坏 | 重新跑 Step 4，加 `--force-overwrite true` |
| decode OOM | seq_len 过大 | 减小 `--seq_lengths`，V100 建议不超过 256 |
