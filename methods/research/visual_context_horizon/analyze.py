"""Aggregate paired E2E, natural-prefix, correctness and quality evidence."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean

from vlm_dflash.eval.native import bootstrap_ci, digest, save, stage_order

from .run_visual_sweep import CONFIG


def common_samples(groups, budgets, samples, repeats, length):
    return [
        sid for sid in samples
        if all(len(groups[budget, sid, repeat, arm]["token_commit_ns"]) >= length
               for budget in budgets for repeat in range(repeats) for arm in ("ar", "sd"))
    ]


def ratios(groups, budget, samples, repeats, horizon, domain):
    times = {arm: {} for arm in ("ar", "sd")}
    for sid in samples:
        for arm, values in times.items():
            cluster = []
            for repeat in range(repeats):
                row = groups[budget, sid, repeat, arm]
                commits = row["token_commit_ns"]
                if horizon is None:
                    timing = row["wall_seconds"] if domain == "request" else (commits[-1] - commits[0]) / 1e9
                elif domain == "request":
                    timing = commits[horizon - 1] / 1e9
                else:
                    timing = (commits[horizon - 1] - commits[0]) / 1e9
                cluster.append(timing)
            values[sid] = cluster
    ar, sd = times["ar"], times["sd"]
    if not samples:
        return dict(n=0)
    total_ar, total_sd = (sum(sum(values) for values in arm.values()) for arm in (ar, sd))
    row = dict(n=len(samples), ar_seconds=total_ar, sd_seconds=total_sd,
               ar_mean_ms=1000 * total_ar / (len(samples) * repeats),
               sd_mean_ms=1000 * total_sd / (len(samples) * repeats))
    if total_ar > 0 and total_sd > 0:
        row["speedup"] = total_ar / total_sd
        if all(sum(values) > 0 for values in sd.values()):
            row["ci95_low"], row["ci95_high"] = bootstrap_ci(ar, sd)
    return row


def load(path):
    return json.loads(path.read_text())


def percentile(values, proportion):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * proportion
    idx = int(rank)
    return (values[idx] + (rank - idx) * (values[min(idx + 1, len(values) - 1)] - values[idx])) / 1e6


def collect(directory, sweep):
    groups = {}
    build_identity = None
    for budget in sweep["budgets"]:
        result = load(directory / f"v{budget}.json")
        if result["status"] != "complete" or result.get("method") != "dflash" or not result.get("resources_verified"):
            raise ValueError(f"Budget {budget} lacks admitted native evidence")
        evidence = directory / f"v{budget}.evidence"
        plan_path = evidence / "plan.json"
        if digest(plan_path) != result["source_plan_sha256"]:
            raise ValueError("Native plan SHA mismatch")
        plan = load(plan_path)
        with (evidence / "resources.jsonl").open() as stream:
            gpu_samples = [json.loads(line) for line in stream if line.strip()]
        pinned_files = {name: sha for name, sha in plan["files"].items()
                        if name not in {item["image"] for item in plan["sessions"]}}
        target_uri = Path(plan["target"])
        target_files = {name: sha for name, sha in plan["files"].items()
                        if Path(name).parent == target_uri or name in plan.get("target_files", [])}
        if plan.get("visual_context_trace") != sweep["trace"]:
            raise ValueError("Native trace mode differs from sweep manifest")
        for stage in stage_order(sweep["repeats"]):
            arm, repeat = stage.split("_r")
            report = load(evidence / f"{stage}.json")
            identity = (pinned_files, report["gpu_uuids"], report["versions"])
            if build_identity is None:
                build_identity = identity
            if identity != build_identity or report["status"] != "complete":
                raise ValueError("Mixed native GPU/runtime identity or incomplete stage")
            timed = [row for row in report["rows"] if row["phase"] == "timed"]
            first = min(row["began_wall_ns"] for row in timed)
            last = max(row["ended_wall_ns"] for row in timed)
            uuid = report["gpu_uuid"]
            sampled_peak = max((reading["per_gpu_memory_mib"][uuid] for reading in gpu_samples
                                if reading["phase"] == "sample" and uuid in reading.get("per_gpu_memory_mib", {})
                                and first <= reading["began_wall_ns"] and reading["ended_wall_ns"] <= last),
                               default=None)
            for row in timed:
                sample = sweep["samples"][row["id"]]
                if row["image_sha256"] != sample["image_sha256"] or row["messages"][-1]["content"][-1]["text"] != sample["question"]:
                    raise ValueError("Sample image or prompt differs from sweep manifest")
                commits = row["token_commit_ns"]
                if not commits or len(commits) != len(row["tokens"]):
                    raise ValueError("Missing native commit timing")
                gaps = [b - a for a, b in zip(commits, commits[1:])]
                row.update(
                    visual_budget=budget, min_pixels=plan["min_pixels"], max_pixels=plan["max_pixels"],
                    mode=arm, repeat=int(repeat), seed=plan["seed"], prompt_hash=sample["prompt_hash"],
                    rendered_prompt_hash=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    dataset_id=sample["dataset_id"], dataset_split=sample["dataset_split"], task=sample["task"],
                    target_identity=target_files, target_model_id=sweep["target_model_id"],
                    target_plan_sha256=result["source_plan_sha256"],
                    drafter_identity=sweep["drafter_checkpoint_sha256"], runtime=report["versions"],
                    actual_attention=report["worker_before"]["attention"], output_length=len(commits),
                    ttft_ms=commits[0] / 1e6, tpot_ms=mean(gaps) / 1e6 if gaps else None,
                    itl_p50_ms=percentile(gaps, .5), itl_p95_ms=percentile(gaps, .95),
                    tokens_per_second=len(commits) / row["wall_seconds"],
                    gpu_uuid=report["gpu_uuid"],
                    gpu_stage_sampled_peak_memory_mib=sampled_peak,
                )
                key = (budget, row["id"], int(repeat), arm)
                if key in groups:
                    raise ValueError(f"Duplicate measured row: {key}")
                groups[key] = row
    for budget in sweep["budgets"]:
        for sid in sweep["samples"]:
            visual = set()
            for repeat in range(sweep["repeats"]):
                ar, sd = (groups[budget, sid, repeat, arm] for arm in ("ar", "sd"))
                if ar["image_grid_thw"] != sd["image_grid_thw"] or ar["prompt_token_ids"] != sd["prompt_token_ids"]:
                    raise ValueError(f"AR/SD conditioning differs: {budget}, {sid}")
                visual.add(tuple(ar["image_grid_thw"]))
                diverged = next((i for i, pair in enumerate(zip(ar["tokens"], sd["tokens"])) if pair[0] != pair[1]),
                                min(len(ar["tokens"]), len(sd["tokens"])))
                for row in (ar, sd):
                    row["token_exact"] = ar["tokens"] == sd["tokens"]
                    row["first_divergence_zero_based"] = None if row["token_exact"] else diverged
            if len(visual) != 1:
                raise ValueError(f"Native grid differs across repeats: {budget}, {sid}")
    for sid in sweep["samples"]:
        counts = [groups[budget, sid, 0, "ar"]["visual_tokens"] for budget in sorted(sweep["budgets"])]
        if any(left >= right for left, right in zip(counts, counts[1:])):
            raise ValueError(f"Actual native vision grid did not grow across budgets: {sid}, {counts}")
    return groups


def score(task, text, answers):
    from research.qwen3vl8b_native_vllm.quality_eval_v1.scoring import score as reference_score
    if task != "general_vqa":
        return reference_score(task, text, answers)
    from research.qwen3vl8b_native_vllm.quality_eval_v1.scoring import EvalAIAnswerProcessor

    class VQAv2AnswerProcessor(EvalAIAnswerProcessor):
        COMMA_STRIP = re.compile(r"(\d)(,)(\d)")

    processor = VQAv2AnswerProcessor()
    if len(answers) != 10:
        raise ValueError("VQAv2 requires exactly ten official answers")
    def clean(answer):
        return answer.replace("\n", " ").replace("\t", " ").strip()
    prediction, gold = clean(text), [clean(answer) for answer in answers]
    if len(set(gold)) > 1:
        def normalize(answer):
            return processor.process_digit_article(processor.process_punctuation(answer))
        prediction, gold = normalize(prediction), [normalize(answer) for answer in gold]
    matches = sum(answer == prediction for answer in gold)
    return mean(min(1, (matches - (answer == prediction)) / 3) for answer in gold)


def summarize(sweep, groups):
    budgets, repeats, samples = sweep["budgets"], sweep["repeats"], sweep["samples"]
    scopes = {name: [sid for sid, value in samples.items() if value["task"] == name]
              for name in sorted({value["task"] for value in samples.values()})}
    scopes["all"] = list(samples)
    summary, break_even = [], []
    for task, ids in scopes.items():
        eligibility = {length: common_samples(groups, budgets, ids, repeats, length) for length in CONFIG["horizons"]}
        for budget in budgets:
            measured = {}
            for horizon, chosen in [(None, ids), *eligibility.items()]:
                for domain in ("request", "decode"):
                    actual = [groups[budget, sid, 0, "ar"]["visual_tokens"] for sid in chosen]
                    matched = [groups[budget, sid, rep, "ar"]["token_exact"]
                               for sid in chosen for rep in range(repeats)]
                    row = dict(task=task, budget=budget, horizon=horizon, domain=domain,
                               visual_tokens_min=min(actual, default=None),
                               visual_tokens_mean=mean(actual) if actual else None,
                               visual_tokens_max=max(actual, default=None),
                               token_exact_fraction=mean(matched) if matched else None)
                    row.update(ratios(groups, budget, chosen, repeats, horizon, domain))
                    if horizon is None and domain == "request" and task not in {"all", "long_form"}:
                        for arm in ("ar", "sd"):
                            row[f"{arm}_quality"] = mean(
                                score(task, groups[budget, sid, rep, arm]["text"], samples[sid]["answers"])
                                for sid in ids for rep in range(repeats)
                            )
                        row["ar_quality_delta_vs_highest"] = row["ar_quality"] - mean(
                            score(task, groups[max(budgets), sid, rep, "ar"]["text"], samples[sid]["answers"])
                            for sid in ids for rep in range(repeats)
                        )
                    summary.append(row)
                    if horizon is not None and domain == "request":
                        measured[horizon] = row.get("speedup")
            crossing = next((length for length, speedup in measured.items() if speedup is not None and speedup > 1), None)
            measured_horizons = [length for length, speedup in measured.items() if speedup is not None]
            status = "observed" if crossing is not None else "none_in_measured_horizons" if measured_horizons else "no_eligible_samples"
            break_even.append(dict(task=task, budget=budget, horizon=crossing, status=status,
                                   measured_horizons=measured_horizons))
    return summary, break_even


def render_report(sweep, rows, breaks):
    phase_names = {"smoke": "GPU smoke", "pilot": "Pilot", "full": "正式实验"}
    phase = phase_names[sweep["phase"]]
    tasks = sorted({row["task"] for row in rows if row["task"] != "all"})
    def speed(row):
        value = row.get("speedup")
        if value is None:
            return "—"
        ci = (row.get("ci95_low"), row.get("ci95_high"))
        return f"{value:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}]" if all(x is not None for x in ci) else f"{value:.3f}×"

    sections = [f"# {phase} · Visual Input Budget × Generation Horizon\n",
                f"使用 {sweep['target_model_id']} 和空间 DFlash B8，计时重复 {sweep['repeats']} 次。"
                "加速比按配对样本的总 AR 时间除以总 SD 时间计算，方括号给出样本级配对 bootstrap 的 95% 区间。",
                "## 自然结束请求\n",
                "| 任务 | 预算 | N | 实测视觉 token | E2E 加速 | Decode-only 加速 | Token 一致率 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    natural = {(row["task"], row["budget"], row["domain"]): row for row in rows if row["horizon"] is None}
    for task in tasks:
        for budget in sweep["budgets"]:
            row = natural[task, budget, "request"]
            decode = natural[task, budget, "decode"]
            sections.append(f"| {task} | {budget} | {row['n']} | {row['visual_tokens_min']}–{row['visual_tokens_max']} | "
                            f"{speed(row)} | {speed(decode)} | {row['token_exact_fraction']:.1%} |")
    sections.extend(["", "## 生成前缀盈亏点\n",
                     "最短盈亏点取离散测量点上 SD 快于 AR 的首个输出前缀长度。"
                     "每个长度使用所有视觉预算、模式和重复都能达到的共同样本。不同长度的样本集可以缩小。", "",
                     "| 任务 | 预算 | 首个测量盈亏点 | 各长度的有效 N |",
                     "| --- | ---: | ---: | --- |"])
    prefixes = {(row["task"], row["budget"], row["horizon"]): row for row in rows
                if row["horizon"] is not None and row["domain"] == "request"}
    status_text = {"no_eligible_samples": "无合格样本", "none_in_measured_horizons": "实测范围内未观察到"}
    for item in breaks:
        task, vb = item["task"], item["budget"]
        if task == "all":
            continue
        counts = " / ".join(f"{length}={prefixes[task, vb, length]['n']}" for length in CONFIG["horizons"])
        sections.append(f"| {task} | {vb} | {item['horizon'] or status_text[item['status']]} | {counts} |")
    quality = [row for (task, _, domain), row in natural.items() if domain == "request" and "ar_quality" in row]
    if quality:
        sections.extend(["", "## QA 质量\n",
                         "| 任务 | 预算 | AR | SD | AR 相对最高预算变化 |",
                         "| --- | ---: | ---: | ---: | ---: |"])
        for row in quality:
            sections.append(f"| {row['task']} | {row['budget']} | {row['ar_quality']:.3f} | "
                            f"{row['sd_quality']:.3f} | {row['ar_quality_delta_vs_highest']:+.3f} |")
    sections.extend(["", "各视觉预算会改变输入分辨率和可见细节。长回答任务的延迟独立汇报。"
                     "本轮仅使用 DFlash，跨 drafter 结论需要单独的实验。"])
    return "\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sweep = load(args.run / "sweep.json")
    if sweep["dry_run"] or sweep["trace"] != "timing":
        parser.error("Headline analysis requires native timing evidence")
    groups = collect(args.run, sweep)
    rows, breaks = summarize(sweep, groups)
    args.out.mkdir(parents=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (args.out / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save(args.out / "break_even.json", breaks)
    (args.out / "REPORT.md").write_text(render_report(sweep, rows, breaks))
    with (args.out / "raw.jsonl").open("x") as stream:
        for row in groups.values():
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
