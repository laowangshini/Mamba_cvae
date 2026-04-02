#!/usr/bin/env python3
"""
为 Phase3 预计算 CelebA 每张图对应的 CLIP 文本 Token 隐藏状态（last_hidden_state），
保存为 .pt 供 train.py / evaluate.py 的 cond_mode=clip_seq 使用。

文本由 list_attr_celeba 中值为 1 的属性名拼接而成（与 Phase2 语义一致）。
默认使用 openai/clip-vit-large-patch14（文本隐层 768，与 1.txt 一致）。
若你已提前下载好模型到本地目录（例如 /root/autodl-tmp/CLIP），可直接 --model 指向该目录，
脚本会自动启用 local_files_only=True，避免联网。

用法示例：
  python scripts/precompute_celeba_clip_text_seq.py \\
    --out data/celeba/celeba_clip_text_vitl14.pt --limit 60000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _list_images(img_dir: str) -> list[str]:
    return sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )


def attrs_to_prompt(attr_names: list[str], vals: list[str]) -> str:
    parts: list[str] = []
    for name, v in zip(attr_names, vals):
        try:
            fv = float(v)
        except ValueError:
            continue
        if fv > 0:
            parts.append(name.replace("_", " ").lower())
    if not parts:
        return "a face portrait"
    return "a face with " + ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLIP text sequences for CelebA")
    parser.add_argument(
        "--img-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "celeba", "img_align_celeba"),
    )
    parser.add_argument(
        "--attr-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "celeba", "list_attr_celeba.csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出 .pt 路径（含 per_image 字典）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace CLIP 文本塔；large-patch14 为 768 维",
    )
    parser.add_argument("--limit", type=int, default=60_000, help="按排序文件名取前 N 张")
    parser.add_argument("--cond-dim", type=int, default=40, help="属性列数（CelebA 默认 40）")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        print("需要安装 transformers: pip install transformers", file=sys.stderr)
        raise SystemExit(1) from e

    if not os.path.isdir(args.img_dir):
        raise SystemExit(f"图像目录不存在: {args.img_dir}")
    if not os.path.isfile(args.attr_csv):
        raise SystemExit(f"属性 CSV 不存在: {args.attr_csv}")

    rows: dict[str, list[str]] = {}
    attr_names: list[str] = []
    with open(args.attr_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        attr_names = header[1 : 1 + args.cond_dim]
        for row in reader:
            if len(row) < 1 + args.cond_dim:
                continue
            rows[row[0].strip()] = row[1 : 1 + args.cond_dim]

    names = _list_images(args.img_dir)[: args.limit]

    local_only = os.path.isdir(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model, local_files_only=local_only)
    text_model = CLIPTextModel.from_pretrained(
        args.model, local_files_only=local_only, use_safetensors=local_only
    )
    text_model.eval()
    text_model.to(device)

    hidden = text_model.config.hidden_size
    seq_len = text_model.config.max_position_embeddings

    per_image: dict[str, torch.Tensor] = {}
    prompt_per_image: dict[str, str] = {}
    missing_attr = 0

    with torch.no_grad():
        for name in tqdm(names, desc="CLIP text encode"):
            vals = rows.get(name)
            if vals is None:
                missing_attr += 1
                continue
            text = attrs_to_prompt(attr_names, vals)
            prompt_per_image[name] = text
            inputs = tokenizer(
                [text],
                padding="max_length",
                max_length=seq_len,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = text_model(**inputs)
            h = out.last_hidden_state.squeeze(0).float().cpu()
            per_image[name] = h.half()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(PROJECT_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload = {
        "per_image": per_image,
        "prompt_per_image": prompt_per_image,
        "clip_text_dim": hidden,
        "seq_len": seq_len,
        "model_name": args.model,
        "limit": args.limit,
        "missing_attr_files": missing_attr,
    }
    torch.save(payload, out_path)
    print(
        f"Saved {len(per_image)} entries -> {out_path} "
        f"(clip_text_dim={hidden}, seq_len={seq_len}, missing_attr={missing_attr})"
    )


if __name__ == "__main__":
    main()
