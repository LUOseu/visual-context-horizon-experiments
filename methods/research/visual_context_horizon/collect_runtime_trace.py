"""Collect native vLLM output events without modifying scheduler or sampler decisions."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import prod
import time


def commit_times(events, expected, start_ns, end_ns):
    observed = []
    times = []
    previous = start_ns
    for tokens, stamp in events:
        if not previous <= stamp <= end_ns:
            raise ValueError("Nonmonotonic or out-of-request commit timestamp")
        observed.extend(tokens)
        times.extend([stamp - start_ns] * len(tokens))
        previous = stamp
    if observed != expected:
        raise ValueError("Engine commit stream disagrees with public output IDs")
    return times


def visual_count(grid, merge, prompt_ids, image_token_id):
    if len(grid) != 3 or grid[0] != 1 or any(d <= 0 for d in grid):
        raise ValueError("Expected a single Qwen3-VL image grid")
    if merge <= 0 or any(d % merge for d in grid[1:]):
        raise ValueError("Visual grid cannot be merged exactly")
    count = prod(grid) // merge**2
    if prompt_ids.count(image_token_id) != count:
        raise ValueError("Visual grid disagrees with prompt placeholder count")
    return count


@dataclass
class Capture:
    events: list = field(default_factory=list)
    features: list = field(default_factory=list)
    render_ns: int | None = None
    submit_ns: int | None = None
    request_id: str | None = None
    foreign_requests: bool = False
    verification_rounds: list = field(default_factory=list)
    kv_usage: list = field(default_factory=list)


@contextmanager
def observe(llm, diagnostic=False):
    engine = llm.llm_engine
    core = engine.engine_core
    output_processor = engine.output_processor
    capture = Capture()

    original_add = output_processor.add_request
    original_get = core.get_output
    original_render = llm.renderer.render_cmpl

    def add(request, *args, **kwargs):
        if capture.request_id is not None:
            raise ValueError("Expected exactly one active native request")
        capture.request_id = request.request_id
        capture.features = request.mm_features
        if diagnostic:
            capture.submit_ns = time.perf_counter_ns()
        return original_add(request, *args, **kwargs)

    def get():
        output = original_get()
        if output.outputs:
            arrived = time.perf_counter_ns()
            for event in output.outputs:
                if event.request_id != capture.request_id:
                    capture.foreign_requests = True
                elif event.new_token_ids:
                    capture.events.append((event.new_token_ids, arrived))
            if diagnostic and output.scheduler_stats is not None:
                stats = output.scheduler_stats
                capture.kv_usage.append(stats.kv_cache_usage)
                spec = stats.spec_decoding_stats
                if spec is not None:
                    if spec.num_drafts != 1 or len(output.outputs) != 1:
                        raise ValueError("Expected one scheduler verification round per diagnostic event")
                    proposed = spec.num_draft_tokens
                    accepted = spec.num_accepted_tokens
                    capture.verification_rounds.append(dict(
                        proposed=proposed, accepted_draft_tokens=accepted,
                        accepted_by_slot=spec.num_accepted_tokens_per_pos,
                        discarded_proposal_tokens=proposed - accepted,
                        first_rejection_slot=accepted if accepted < proposed else None,
                        committed_bonus_tokens=int(accepted == proposed and len(output.outputs[0].new_token_ids) > accepted),
                    ))
        return output

    def render(*args, **kwargs):
        started = time.perf_counter_ns()
        result = original_render(*args, **kwargs)
        capture.render_ns = time.perf_counter_ns() - started
        return result

    output_processor.add_request = add
    core.get_output = get
    if diagnostic:
        llm.renderer.render_cmpl = render
    try:
        yield capture
    finally:
        output_processor.add_request = original_add
        core.get_output = original_get
        if diagnostic:
            llm.renderer.render_cmpl = original_render


def row_trace(capture, processor, row, width, height, diagnostic):
    if capture.foreign_requests or not capture.features or len(capture.features) != 1:
        raise ValueError("Native trace requires exactly one request and one image feature")
    feature = capture.features[0]
    if feature.modality != "image":
        raise ValueError("Native feature is not an image")
    grid = feature.data["image_grid_thw"].data.tolist()
    merge = processor.image_processor.merge_size
    patch = processor.image_processor.patch_size
    count = visual_count(grid, merge, row["prompt_token_ids"], processor.image_token_id)
    return {
        "token_commit_ns": commit_times(capture.events, row["tokens"], row["began_ns"], row["ended_ns"]),
        "image_width": width,
        "image_height": height,
        "image_pixels": width * height,
        "image_grid_thw": grid,
        "image_patch_size": patch,
        "image_merge_size": merge,
        "processed_width": grid[2] * patch,
        "processed_height": grid[1] * patch,
        "visual_tokens": count,
        "text_prompt_tokens": len(row["prompt_token_ids"]) - count,
        "context_tokens": len(row["prompt_token_ids"]),
        "diagnostic": {
            "renderer_and_mm_preprocess_ns": capture.render_ns,
            "submit_offset_ns": capture.submit_ns - row["began_ns"],
            "verification_rounds": capture.verification_rounds,
            "max_scheduler_kv_usage_fraction": max(capture.kv_usage, default=None),
        } if diagnostic else None,
    }



class WorkerStageTimers:
    """Request-scoped CUDA events for the separate diagnostic lane."""
    METHODS = {
        "_execute_mm_encoder": "vision_encoder",
        "_model_forward": "target_forward",
        "_sample": "sampler",
        "propose_draft_token_ids": "draft_proposer",
        "_preprocess": "runner_preprocess_inclusive",
    }

    def __init__(self, runner, torch):
        self.runner = runner
        self.torch = torch
        self.events = {}
        self.original = []

    def attach(self):
        if self.original:
            raise ValueError("Diagnostic worker timers already enabled")
        for name, label in self.METHODS.items():
            self._attach(self.runner, name, label)
        self._attach(self.runner.model, "compute_logits", "target_lm_head")
        drafter = getattr(self.runner, "drafter", None)
        if drafter is not None:
            self._attach(drafter.model, "precompute_and_store_context_kv", "draft_context_kv")
            self._attach(drafter.model, "forward", "draft_forward")

    def _attach(self, item, name, label):
        original = getattr(item, name)
        self.original.append((item, name, original))
        setattr(item, name, self._timed(original, label))

    def _timed(self, method, label):
        def wrapper(*args, **kwargs):
            first, last = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
            first.record()
            result = method(*args, **kwargs)
            last.record()
            self.events.setdefault(label, []).append((first, last))
            return result
        return wrapper

    def detach(self):
        for item, name, method in reversed(self.original):
            setattr(item, name, method)
        self.original.clear()

    def read(self):
        return {label: [start.elapsed_time(end) for start, end in events]
                for label, events in self.events.items()}
