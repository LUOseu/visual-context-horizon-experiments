# Visual Input Budget × Generation Horizon

The experiment uses Qwen3-VL-8B-Instruct with the VelaDraft spatial DFlash B8 checkpoint in native vLLM 0.22.1. AR and SD run in separate fresh batch-one, TP1 BF16 processes. The target uses deterministic greedy decoding, natural EOS, a 512-token safety cap, and disabled prefix and multimodal processor caching.

## Runtime admission

The original experiment documents are [support matrix](../../SUPPORT_MATRIX.md), [benchmark evidence](../../BENCHMARKS.md), [native protocol](../../docs/benchmark.md) and [Qwen3-VL-8B snapshot](../qwen3vl8b_native_vllm/README.md). The snapshot records the A6000 protocol. A different machine needs its own smoke and resource evidence.

The DFlash adapter preserves the trained drafter and three-axis spatial positions. Admission requires the pinned upstream standard greedy verifier and native worker evidence. In standard greedy mode the verifier commits target argmax tokens up to the first disagreement and may emit a target bonus after complete acceptance. AR/SD token equality is measured and reported as a BF16 runtime diagnostic, including the first differing position and output length. An observed numerical mismatch does not itself fail admission.

Example commands run from the VelaDraft checkout. Use an already provisioned Linux CUDA environment with vLLM 0.22.1, PyTorch 2.11.0, and Transformers 5.11.0. Use the pinned target revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and the historical 75,000-step checkpoint.

```bash
python research/qwen3vl8b_native_vllm/spatial_adapter/export_checkpoint.py "$CHECKPOINT" "$DRAFT"
python -m research.visual_context_horizon.prepare_runtime "$CHECKPOINT" "$RUNTIME"
python -m research.visual_context_horizon.run_visual_sweep \
  --target "$TARGET" --draft "$DRAFT" --runtime-manifest "$RUNTIME" \
  --source "$WORKLOADS" --phase smoke --count 2 --budgets 256 1280 --repeats 1 \
  --gpu 0 --out "$RESULTS/smoke"
```

Read the smoke evidence before running `--phase pilot` with 24 requests per task, followed by `--phase full` with 100 per task. Both stages use all five budgets and at least three repeated pairs. The single-turn reference checks canonical inputs with a one-token cap. Every measured input is warmed before timing and two long-form warmups use the 512-token cap. The other warmups use a 16-token cap. Timed Triton JIT invalidates that run. A clean run with full-length warmups is available via `--full-warmup`. Keep separate output directories. Use `--diagnostic` with a small smoke scope to capture native worker profiles, renderer preparation, worker allocator peaks, and native acceptance counters. Profiling timings describe instrumented runs and live in their own evidence directory.

## Workloads

- TextVQA, DocVQA and ChartQA reuse the archived 256-question real-gold datasets and original image bytes. `restore_frozen_qa` reconstructs missing images from pinned Parquet shards, checking both shard and selected-image hashes.
- General VQA uses VQAv2 validation annotations. The selection picks one question per COCO image and uses all ten human answers.
- Long-form generation uses COCO val2017 and a fixed descriptive prompt. It contributes to generation timing with its own task label and has no QA score.
- `prepare_coco_workloads` can read the official COCO val2017 ZIP or its central directory and individually downloaded JPEGs that match the ZIP CRC and file size.
- `prepare_workloads` writes a frozen manifest from explicit task directories. The experiment records sample, question and image identity. An incomplete local cache can be explicitly marked `--smoke-only`.

## Measurement

Each input budget applies the official processor's minimum and maximum pixel area bounds. The maximum is the desired count times `32²` for 16-pixel patches and a 2×2 spatial merge. The minimum is 85% of the desired count, rounded to an integer token-equivalent area. Processor rounding depends on image aspect ratio. The analysis uses the actual vLLM image grid and cross-checks the number of visual placeholder IDs. Visual context budget changes the processed image and thus can affect task accuracy.

The request wall timer starts before image read and decode and ends after the native request and CUDA synchronization. Native EngineCore delivery timestamps attach to the actual output IDs and share a timestamp for tokens committed together. TTFT measures elapsed wall time to parent EngineCore delivery of the first token. The timing lane's observer adds a timestamp and list entry per delivery event. AR can have more delivery events than SD, which matters when interpreting marginal speed differences. Output lengths include a committed terminal token when vLLM returns it. Decode-only measures first-to-last observed commit time. Direct comparison of AR and SD generation horizon uses the same images whose natural outputs reached that horizon across all budgets, arms and repeats. All natural outputs remain in the full-request analysis.

The primary ratio uses summed AR latency divided by summed SD latency. The sample-level paired 95% bootstrap resamples each image's repeats together. The break-even result is the first sampled horizon whose point estimate exceeds 1×. Unobserved horizons retain their sample counts. The eligible cohort can shrink across horizons, so the crossing describes the measured grid. GPU memory in the timing run is sampled device usage within each timed stage. The diagnostic lane collects worker process allocator peaks. Verifier acceptance counts precede scheduler clipping at EOS or the token cap, and bonus counts are computed from committed output.

```bash
python -m research.visual_context_horizon.run_horizon_analysis "$RESULTS/full" --out "$RESULTS/analysis"
python -m research.visual_context_horizon.plot_results "$RESULTS/analysis" --out "$RESULTS/figures"
```

Primary-source references for the processor and verifier are the [Transformers image processor](https://github.com/huggingface/transformers/blob/v5.11.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py), [vLLM Qwen3-VL integration](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/models/qwen3_vl.py) and [vLLM rejection sampler](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/sample/rejection_sampler.py). Release data sources are the [VQA download](https://visualqa.org/download) and [COCO download](https://cocodataset.org/#download).
