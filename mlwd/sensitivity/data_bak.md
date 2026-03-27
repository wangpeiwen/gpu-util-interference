



➜  gpu-util-interference git:(main) ✗ PYTHONPATH=. python3 -m mlwd.collect_all --model /data/Qwen/Qwen2.5-7B-Instruct --stage sensitivity --batch_sizes 1 --seq_lengths 32        
MLWD Offline Collection
  Models: ['/data/Qwen/Qwen2.5-7B-Instruct']
  Quantizations: ['fp16']
  TP degrees: [1]
  Batch sizes: [1]
  Seq lengths: [32]
  Stages: {'sensitivity'}
  Total experiment points: 2
  DB: mlwd_data.db

============================================================
Config: /data/Qwen/Qwen2.5-7B-Instruct | fp16 | TP=1 (config_id=1)

--- Stage 3: Interference Sensitivity ---
  [Sensitivity] Loading model: /data/Qwen/Qwen2.5-7B-Instruct...
INFO 03-27 12:18:25 [utils.py:233] non-default args: {'trust_remote_code': True, 'dtype': 'float16', 'disable_log_stats': True, 'enforce_eager': True, 'model': '/data/Qwen/Qwen2.5-7B-Instruct'}
WARNING 03-27 12:18:25 [envs.py:1717] Unknown vLLM environment variable detected: VLLM_USE_V1
INFO 03-27 12:18:25 [model.py:533] Resolved architecture: Qwen2ForCausalLM
WARNING 03-27 12:18:25 [model.py:1920] Casting torch.bfloat16 to torch.float16.
INFO 03-27 12:18:25 [model.py:1582] Using max model len 32768
INFO 03-27 12:18:25 [scheduler.py:231] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 03-27 12:18:25 [vllm.py:754] Asynchronous scheduling is enabled.
WARNING 03-27 12:18:25 [vllm.py:788] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 03-27 12:18:25 [vllm.py:799] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 03-27 12:18:26 [vllm.py:964] Cudagraph is disabled under eager mode
INFO 03-27 12:18:26 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=9011) INFO 03-27 12:18:27 [core.py:103] Initializing a V1 LLM engine (v0.18.0) with config: model='/data/Qwen/Qwen2.5-7B-Instruct', speculative_config=None, tokenizer='/data/Qwen/Qwen2.5-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/data/Qwen/Qwen2.5-7B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=9011) INFO 03-27 12:18:28 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://10.244.29.174:50369 backend=nccl
(EngineCore pid=9011) INFO 03-27 12:18:28 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=9011) INFO 03-27 12:18:32 [gpu_model_runner.py:4481] Starting to load model /data/Qwen/Qwen2.5-7B-Instruct...
(EngineCore pid=9011) ERROR 03-27 12:18:33 [fa_utils.py:145] Cannot use FA version 2 is not supported due to FA2 is only supported on devices with compute capability >= 8
(EngineCore pid=9011) INFO 03-27 12:18:34 [cuda.py:317] Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION'].
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [01:25<04:15, 85.31s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [02:55<02:56, 88.07s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [04:31<01:31, 91.68s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [05:43<00:00, 83.87s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [05:43<00:00, 85.80s/it]
(EngineCore pid=9011) 
(EngineCore pid=9011) INFO 03-27 12:24:18 [default_loader.py:384] Loading weights took 344.06 seconds
(EngineCore pid=9011) INFO 03-27 12:24:19 [gpu_model_runner.py:4566] Model loading took 14.25 GiB memory and 345.277135 seconds
(EngineCore pid=9011) INFO 03-27 12:24:24 [gpu_worker.py:456] Available KV cache memory: 13.1 GiB
(EngineCore pid=9011) INFO 03-27 12:24:24 [kv_cache_utils.py:1316] GPU KV cache size: 245,360 tokens
(EngineCore pid=9011) INFO 03-27 12:24:24 [kv_cache_utils.py:1321] Maximum concurrency for 32,768 tokens per request: 7.49x
(EngineCore pid=9011) INFO 03-27 12:24:25 [core.py:281] init engine (profile, create kv cache, warmup model) took 6.20 seconds
(EngineCore pid=9011) WARNING 03-27 12:24:26 [vllm.py:788] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=9011) WARNING 03-27 12:24:26 [vllm.py:799] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=9011) INFO 03-27 12:24:26 [vllm.py:964] Cudagraph is disabled under eager mode
INFO 03-27 12:24:26 [llm.py:391] Supported tasks: ['generate']
  [Sensitivity] Model loaded.

  [Sensitivity] Collecting: b=1, s=32, prefill
  [Sensitivity] Measuring baseline (no interference)...
Rendering prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 30.34it/s]
Processed prompts: 100%|███████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.18it/s, est. speed input: 101.91 toks/s, output: 3.18 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1233.62it/s]
Processed prompts: 100%|███████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.19it/s, est. speed input: 166.66 toks/s, output: 5.21 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1432.48it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.88it/s, est. speed input: 908.10 toks/s, output: 28.35 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 865.70it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.99it/s, est. speed input: 916.96 toks/s, output: 28.62 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 663.24it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 28.28it/s, est. speed input: 927.74 toks/s, output: 28.95 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 656.69it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 29.47it/s, est. speed input: 951.16 toks/s, output: 29.71 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1515.28it/s]
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 31.17it/s, est. speed input: 1006.01 toks/s, output: 31.42 toks/s]
  [Sensitivity] Baseline: 38.85 ms
  [Sensitivity] Measuring σ_bs...
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1930.19it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 15.78it/s, est. speed input: 507.10 toks/s, output: 15.84 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1409.85it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 13.95it/s, est. speed input: 451.96 toks/s, output: 14.11 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 660.83it/s]
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 30.61it/s, est. speed input: 1004.17 toks/s, output: 31.34 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 673.03it/s]
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 31.15it/s, est. speed input: 1005.37 toks/s, output: 31.40 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 138.28it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 11.01it/s, est. speed input: 353.18 toks/s, output: 11.03 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2019.40it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][BS Stress] Run 0: Latency: 163.465179 ms
Block Scheduler stress kernel completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 12.35it/s, est. speed input: 396.81 toks/s, output: 12.40 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1931.97it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 11.64it/s, est. speed input: 373.32 toks/s, output: 11.66 toks/s]
[BS Stress] Run 0: Latency: 142.977951 ms
Block Scheduler stress kernel completed!
  [Sensitivity] σ_bs = 1.1370 (baseline=38.85ms, stressed=83.02ms)
  [Sensitivity] Measuring σ_cu...
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2094.01it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.277632 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.353408 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.229760 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.297600 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.342496 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.351040 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345312 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 14.15it/s, est. speed input: 454.22 toks/s, output: 14.19 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2101.35it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.133344 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.339776 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.344128 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.343872 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.227776 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.309600 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 11.87it/s, est. speed input: 380.96 toks/s, output: 11.90 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2167.60it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.136032 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.316064 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.347584 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.347072 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.347392 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345056 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.344896 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 2.797952 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.13it/s, est. speed input: 324.70 toks/s, output: 10.15 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2134.51it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.137664 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.308000 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345504 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.346432 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.347040 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345440 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.341440 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 2.798656 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.09it/s, est. speed input: 323.61 toks/s, output: 10.11 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2120.48it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.139424 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.347680 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.343712 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.346304 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.252800 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345472 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.297568 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.13it/s, est. speed input: 325.02 toks/s, output: 10.16 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2085.68it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.135808 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.309952 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.351008 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345216 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.348800 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.350208 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345344 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.13it/s, est. speed input: 324.87 toks/s, output: 10.15 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2074.33it/s]
Processed prompts:   0%|                                                             | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][FP32] Run 0: Latency: 3.134496 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 4.104032 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 4.788192 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.321152 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.322880 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.350240 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.347520 ms
FP32 Kernel execution completed!
[FP32] Run 0: Latency: 5.345184 ms
FP32 Kernel execution completed!
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.08it/s, est. speed input: 324.37 toks/s, output: 10.13 toks/s]
[FP32] Run 0: Latency: 2.801408 ms
FP32 Kernel execution completed!
  [Sensitivity] σ_cu = 1.5745 (baseline=38.85ms, stressed=100.02ms)
  [Sensitivity] Measuring σ_l2...
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1547.71it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.47it/s, est. speed input: 336.22 toks/s, output: 10.51 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1985.94it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.22it/s, est. speed input: 327.89 toks/s, output: 10.25 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2023.30it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.12it/s, est. speed input: 324.96 toks/s, output: 10.15 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2182.26it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.20it/s, est. speed input: 327.64 toks/s, output: 10.24 toks/s][L2 Stress] Run 0: Latency: 396.946686 ms

L2 Cache stress kernel completed!
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2013.59it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.11it/s, est. speed input: 326.17 toks/s, output: 10.19 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 686.47it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.58it/s, est. speed input: 339.27 toks/s, output: 10.60 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2174.34it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.10it/s, est. speed input: 323.89 toks/s, output: 10.12 toks/s]
[L2 Stress] Run 0: Latency: 366.695435 ms
L2 Cache stress kernel completed!
  [Sensitivity] σ_l2 = 1.5783 (baseline=38.85ms, stressed=100.17ms)
  [Sensitivity] Measuring σ_bw...
Rendering prompts:   0%|                                                                                                                 | 0/1 [00:00<?, ?it/s]Failed: Cuda error /home/zhangry/peipei/gpu-util-interference/mm_pytorch/python_interface.cu:204 'out of memory'
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2059.06it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 22.53it/s, est. speed input: 736.65 toks/s, output: 23.00 toks/s]
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 649.57it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 29.24it/s, est. speed input: 944.63 toks/s, output: 29.50 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1414.61it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 15.50it/s, est. speed input: 497.71 toks/s, output: 15.55 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1544.29it/s]
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 33.98it/s, est. speed input: 1097.81 toks/s, output: 34.29 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1353.00it/s]
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 42.17it/s, est. speed input: 1361.32 toks/s, output: 42.52 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1847.71it/s]
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 33.26it/s, est. speed input: 1074.09 toks/s, output: 33.55 toks/s]
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1799.36it/s]
Processed prompts: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 14.46it/s, est. speed input: 464.71 toks/s, output: 14.52 toks/s]