"""
绘制门控梯度 L2 范数对比折线图（回应 Reviewer 意见 4：门控热启动缺乏实证支撑）。

输入：两个/多个 run 目录的 gate_grad_log.csv（由 train.py 在 gate_grad_log=True 时自动写出）。
输出：1 张 PNG 折线图（含全程 + 早期放大子图）。

用法示例：
  python scripts/plot_gate_grad.py \
      --logs experiments/exp_e_gate_warmstart_init0_*/gate_grad_log.csv:gate_init=0.0 \
             experiments/exp_e_gate_warmstart_init0p02_*/gate_grad_log.csv:gate_init=0.02 \
      --out figs/gate_grad_warmstart.png \
      --gate-idx 0
"""
import argparse
import csv
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log_arg(s):
    if ":" in s:
        path, label = s.split(":", 1)
    else:
        path, label = s, os.path.basename(os.path.dirname(s))
    matches = sorted(glob.glob(path))
    if not matches:
        raise SystemExit(f"no match for {path}")
    return matches[0], label


def load_log(path):
    steps = []
    grad_l2 = []
    value_mean = []
    value_l2 = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"empty log: {path}")
    header = rows[0].keys()
    idx0_grad = "gate0_grad_l2" if "gate0_grad_l2" in header else None
    idx0_val = "gate0_value_mean" if "gate0_value_mean" in header else None
    idx0_vl2 = "gate0_value_l2" if "gate0_value_l2" in header else None
    if idx0_grad is None:
        raise SystemExit(f"log {path} missing gate0_grad_l2 column")
    for r in rows:
        try:
            s = int(r["global_step"])
            steps.append(s)
            grad_l2.append(float(r[idx0_grad]) if r[idx0_grad] not in ("", "nan", "NaN") else float("nan"))
            value_mean.append(
                float(r[idx0_val]) if idx0_val and r[idx0_val] not in ("", "nan", "NaN") else float("nan")
            )
            value_l2.append(
                float(r[idx0_vl2]) if idx0_vl2 and r[idx0_vl2] not in ("", "nan", "NaN") else float("nan")
            )
        except Exception:
            continue
    return steps, grad_l2, value_mean, value_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="每项为 <csv_path[glob]>:<label>，例如 experiments/exp_e_*/gate_grad_log.csv:gate_init=0.02",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--gate-idx", type=int, default=0, help="读取第几层 GatedCrossAttn 的门控（0/1/2）")
    parser.add_argument("--smooth", type=int, default=10, help="对梯度做窗口平滑的窗口大小（>=1）")
    parser.add_argument("--early-steps", type=int, default=200, help="放大子图覆盖的训练步数")
    parser.add_argument("--title", type=str, default="Gate Gradient L2-norm During Training")
    args = parser.parse_args()

    series = []
    for spec in args.logs:
        path, label = parse_log_arg(spec)
        steps, g, v_mean, v_l2 = load_log(path)
        series.append((label, path, steps, g, v_mean, v_l2))

    if args.smooth and args.smooth > 1:
        def smooth(xs, w):
            if w <= 1 or len(xs) <= 1:
                return xs
            ys = []
            buf = []
            for x in xs:
                buf.append(x)
                if len(buf) > w:
                    buf.pop(0)
                ys.append(sum(buf) / len(buf))
            return ys
    else:
        def smooth(xs, w):
            return xs

    # 三联面板：左 = 全程梯度 L2（log）； 中 = 早期梯度 L2 放大； 右 = gate 值演化（线性）
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax1 = axes[0]
    for label, _, steps, g, _, _ in series:
        gs = smooth(g, args.smooth)
        ax1.plot(steps, gs, label=label, linewidth=1.5)
    ax1.set_yscale("symlog", linthresh=1e-6)
    ax1.set_xlabel("global step")
    ax1.set_ylabel(r"$\Vert\,\nabla\,\mathrm{gate}\Vert_2$  (smoothed)")
    ax1.set_title("(a) Gradient L2-norm: full training")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="best", fontsize=9)

    ax2 = axes[1]
    for label, _, steps, g, _, _ in series:
        es = [s for s in steps if s < args.early_steps]
        gs_full = smooth(g, max(1, args.smooth // 2))
        gs = gs_full[: len(es)]
        ax2.plot(es, gs, label=label, linewidth=1.6)
    ax2.set_yscale("symlog", linthresh=1e-6)
    ax2.set_xlabel("global step (early window)")
    ax2.set_ylabel(r"$\Vert\,\nabla\,\mathrm{gate}\Vert_2$  (smoothed)")
    ax2.set_title(f"(b) Gradient L2-norm: first {args.early_steps} steps")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="best", fontsize=9)

    ax3 = axes[2]
    for label, _, steps, _, v_mean, _ in series:
        ax3.plot(steps, v_mean, label=label, linewidth=1.6)
    ax3.set_xlabel("global step")
    ax3.set_ylabel(r"$\mathrm{mean}(\mathrm{gate})$")
    ax3.set_title("(c) Gate value evolution")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="best", fontsize=9)

    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # 兼容旧路径：value 单图
    out_value = os.path.splitext(args.out)[0] + "_value.png"
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for label, _, steps, _, v_mean, _ in series:
        ax.plot(steps, v_mean, label=label, linewidth=1.5)
    ax.set_xlabel("global step")
    ax.set_ylabel("mean(gate)")
    ax.set_title("Gate value evolution")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_value, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_value}")

    # 写出统计摘要 CSV（供论文引用 / 表格使用）
    stats_csv = os.path.splitext(args.out)[0] + "_stats.csv"
    cols = [
        "label",
        "log_path",
        "total_steps",
        "value_init",
        "value_at_240",
        "value_final",
        "grad_l2_step0",
        "grad_l2_mean_first50",
        "grad_l2_mean_first240",
        "grad_l2_mean_overall",
    ]
    with open(stats_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for label, path, steps, g, v_mean, _ in series:
            n = len(g)

            def safe_mean(xs):
                xs = [x for x in xs if x == x]
                return sum(xs) / max(1, len(xs))

            row = [
                label,
                path,
                n,
                v_mean[0] if n > 0 else float("nan"),
                v_mean[min(240, n - 1)] if n > 0 else float("nan"),
                v_mean[-1] if n > 0 else float("nan"),
                g[0] if n > 0 else float("nan"),
                safe_mean(g[:50]),
                safe_mean(g[:240]),
                safe_mean(g),
            ]
            w.writerow(row)
    print(f"Saved: {stats_csv}")


if __name__ == "__main__":
    main()
