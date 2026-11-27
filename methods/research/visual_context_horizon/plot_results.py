"""Render visual-budget figures from admitted native summaries."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .run_visual_sweep import CONFIG


def figure(out, stem):
    def finish(fig):
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return finish


def render(rows, breaks, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    tasks = sorted({row["task"] for row in rows if row["task"] != "all"})
    budgets = sorted({int(row["budget"]) for row in rows})
    colors = dict(zip(tasks, ("#007b85", "#375ba3", "#a45929", "#9c4174", "#5d7328"), strict=False))
    request_rows = [row for row in rows if row["domain"] == "request"]
    full = [row for row in request_rows if not row["horizon"] and row["speedup"]]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for task in tasks:
        points = [row for row in full if row["task"] == task]
        if not points:
            continue
        x = [float(row["visual_tokens_mean"]) for row in points]
        y = [float(row["speedup"]) for row in points]
        ax.plot(x, y, "o-", label=task, color=colors[task])
        if all(row["ci95_low"] and row["ci95_high"] for row in points):
            ax.fill_between(x, [float(row["ci95_low"]) for row in points],
                            [float(row["ci95_high"]) for row in points], color=colors[task], alpha=.13)
    ax.axhline(1, color="#737373", linestyle="--", linewidth=1)
    ax.set(xlabel="Mean observed visual tokens", ylabel="Full-request speedup (AR / SD)")
    ax.legend(frameon=False, ncols=2)
    figure(out, "speedup_by_visual_tokens")(fig)

    horizons = CONFIG["horizons"]
    index = {(row["task"], int(row["budget"]), int(row["horizon"])): row for row in request_rows if row["horizon"]}
    matrices = {task: np.array([[float(index[task, budget, length]["speedup"] or "nan")
                                 for length in horizons] for budget in budgets]) for task in tasks}
    observed = [value for matrix in matrices.values() for value in matrix.ravel() if np.isfinite(value)]
    norm = TwoSlopeNorm(vmin=min([.95, *observed]), vcenter=1, vmax=max([1.05, *observed]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.35 * len(tasks), 4.45), layout="constrained", squeeze=False)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e7e7e7")
    for task, ax in zip(tasks, axes[0], strict=True):
        heatmap = ax.imshow(matrices[task], cmap=cmap, norm=norm, aspect="auto")
        ax.set(title=task, xlabel="Committed output tokens",
               xticks=range(len(horizons)), xticklabels=horizons,
               yticks=range(len(budgets)), yticklabels=budgets)
        for i, budget in enumerate(budgets):
            for j, length in enumerate(horizons):
                record = index[task, budget, length]
                cell = matrices[task][i, j]
                label = f"{cell:.2f}×\nN={record['n']}" if np.isfinite(cell) else f"N={record['n']}"
                color = "white" if np.isfinite(cell) and max(1 - norm(cell), norm(cell)) > .84 else "#202020"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.4, color=color)
    axes[0][0].set_ylabel("Target visual-token budget")
    fig.colorbar(heatmap, ax=axes[0], orientation="horizontal", label="AR / SD", shrink=.48)
    figure(out, "visual_context_horizon_heatmaps")(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4))
    for task in tasks:
        points = [item for item in breaks if item["task"] == task and item["horizon"] is not None]
        if points:
            ax.plot([item["budget"] for item in points], [item["horizon"] for item in points],
                    "o-", label=task, color=colors[task])
    if ax.has_data():
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No observed crossover", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Target visual-token budget", ylabel="First measured speedup above 1× (tokens)",
           xticks=budgets, yticks=horizons, ylim=(0, max(horizons) * 1.05))
    figure(out, "break_even_horizon")(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.1 * len(tasks), 3.8), layout="constrained", squeeze=False)
    for task, ax in zip(tasks, axes[0], strict=True):
        points = [row for row in full if row["task"] == task]
        if points:
            positions = np.arange(len(points))
            for arm, offset, color in [("ar", -.18, "#375ba3"), ("sd", .18, "#007b85")]:
                ax.bar(positions + offset, [float(row[f"{arm}_mean_ms"]) for row in points],
                       width=.33, color=color, label=arm.upper())
            ax.set_xticks(positions, [row["budget"] for row in points], rotation=30)
        ax.set(title=task, xlabel="Visual budget")
    axes[0][0].set_ylabel("Mean full-request latency (ms)")
    axes[0][-1].legend(frameon=False)
    figure(out, "task_comparison")(fig)

    quality_tasks = [task for task in tasks if any(row["task"] == task and row.get("ar_quality") for row in full)]
    if quality_tasks:
        fig, axes = plt.subplots(1, len(quality_tasks), figsize=(3.25 * len(quality_tasks), 3.35), squeeze=False)
        for task, ax in zip(quality_tasks, axes[0], strict=True):
            for arm, marker in [("ar", "o"), ("sd", "s")]:
                points = [(float(row[f"{arm}_mean_ms"]), float(row[f"{arm}_quality"]), row["budget"])
                          for row in full if row["task"] == task]
                front, best = [], -1.0
                for latency, score, budget in sorted(points, key=lambda point: (point[0], -point[1])):
                    if score > best:
                        front.append((latency, score))
                        best = score
                    ax.annotate(budget, (latency, score), xytext=(3, 4), textcoords="offset points", fontsize=7)
                ax.scatter([p[0] for p in points], [p[1] for p in points], marker=marker, label=arm.upper())
                ax.plot(*zip(*front), alpha=.5)
            ax.set(title=task, xlabel="Full-request latency (ms)")
        axes[0][0].set_ylabel("Ground-truth quality score")
        axes[0][-1].legend(frameon=False)
        figure(out, "quality_efficiency_pareto")(fig)


def plot_diagnostics(rows, out):
    measures = [
        ("renderer_and_mm_prepare_ms", "Renderer + multimodal preparation"),
        ("vision_encoder_ms", "Vision encoder"),
        ("initial_target_forward_ms", "Initial target forward"),
        ("later_target_forwards_ms", "Subsequent target forwards"),
        ("sampler_ms", "Target sampler"),
        ("draft_proposer_inclusive_ms", "DFlash proposer (inclusive)"),
    ]
    from statistics import mean
    budgets = sorted({int(row["visual_budget"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), layout="constrained")
    for ax, (field, label) in zip(axes.flat, measures, strict=True):
        for arm, offset, color in [("ar", -.18, "#375ba3"), ("sd", .18, "#007b85")]:
            points = []
            for i, budget in enumerate(budgets):
                timings = [float(row[field]) for row in rows if row["arm"] == arm
                           and int(row["visual_budget"]) == budget and row[field]]
                if timings:
                    points.append((i + offset, mean(timings)))
            if points:
                x, y = zip(*points)
                ax.bar(x, y, width=.33, color=color, label=arm.upper())
        ax.set(title=label, ylabel="Mean ms per request",
               xticks=range(len(budgets)), xticklabels=budgets)
    axes[0][0].legend(frameon=False)
    fig.suptitle("Instrumented native stage timings")
    figure(out, "latency_decomposition")(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-stages", type=Path)
    args = parser.parse_args()
    with (args.summary / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    breaks = json.loads((args.summary / "break_even.json").read_text())
    args.out.mkdir(parents=True)
    render(rows, breaks, args.out)
    if args.diagnostic_stages:
        with args.diagnostic_stages.open() as stream:
            plot_diagnostics(list(csv.DictReader(stream)), args.out)


if __name__ == "__main__":
    main()
