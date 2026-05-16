"""
为论文表格补齐 FID（回应 Reviewer 意见 2：缺失生成指标）。

做法（与论文 4.1 节 FID 评价方法一致）：
  1) 对每个 run 目录的 epoch 20 checkpoint，从标准正态先验 z ~ N(0, I) 采样 num-fake 个 z；
  2) 在 cond_mode=clip_* 时，配套从 CLIP 文本序列缓存中随机抽 num-fake 个文本条件；
  3) 调用 decoder 生成 num-fake 张 64x64 假图；
  4) 从 CelebA 训练目录中按文件名顺序前若干张读取 num-real 张真实图（与训练集分布一致）；
  5) 用 torchmetrics.image.fid.FrechetInceptionDistance(feature=2048, normalize=True) 在 Inception
     特征空间内估计 Fréchet 距离，并把每个 run 的 FID 追加到 CSV 中。

用法（默认 num-fake=num-real=10000）：
  python scripts/eval_fid.py \
      --exp-dirs exp_c_phase3_clip_ss2d_20260331_162425 \
                 exp_c_phase3_clip_hybrid_ss2d_20260402_105550 \
                 exp_c_phase3_clip_hybrid_v3_ss2d_20260402_114252 \
                 exp_d_phase4_clip_seq_lpips_cfg_ss2d_20260402_172413 \
      --checkpoint model_epoch_20.pth \
      --out-csv experiments/fid_compare.csv
"""
import argparse
import csv
import os
import random
import time
import warnings

import torch
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance

import sys

PROJECT_ROOT = "/root/autodl-tmp/Mamba-CVAE"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cvae import MambaCVAE  # noqa: E402

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "celeba", "img_align_celeba")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _list_images(img_dir):
    names = sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    return names


class RealImageDataset(Dataset):
    """从训练集中前 num 张图按文件名顺序读取，给 FID 真实分布使用。"""

    def __init__(self, img_dir, names, img_size):
        self.img_dir = img_dir
        self.names = names
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # [0,1]
            ]
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.names[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def build_model_from_run(run_dir):
    config_path = os.path.join(run_dir, "run_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _m = cfg.get("model", {})
    model = MambaCVAE(
        latent_dim=int(_m.get("latent_dim", 128)),
        block_type=str(_m.get("block_type", "ss2d")),
        cond_dim=0 if str(_m.get("cond_mode", "attr")).startswith("clip_") else int(_m.get("cond_dim", 0)),
        cond_embed_dim=int(_m.get("cond_embed_dim", 256)),
        cond_mode=str(_m.get("cond_mode", "attr")).strip().lower(),
        clip_text_dim=int(_m.get("clip_text_dim", 512)),
        mapper_bidirectional=bool(_m.get("mapper_bidirectional", True)),
        attn_heads=int(_m.get("attn_heads", 4)),
        bottleneck_inject_stages=int(_m.get("bottleneck_inject_stages", 1)),
        gate_init=float(_m.get("gate_init", 0.02)),
    ).to(DEVICE)
    return model, cfg


def load_clip_cache(clip_cache_pt):
    cache_path = (
        clip_cache_pt
        if os.path.isabs(clip_cache_pt)
        else os.path.join(PROJECT_ROOT, clip_cache_pt)
    )
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"CLIP 缓存不存在: {cache_path}")
    try:
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(cache_path, map_location="cpu")
    if isinstance(blob, dict) and "per_image" in blob:
        per_image = blob["per_image"]
    else:
        per_image = blob
    return per_image


def evaluate_one_run(run_folder, checkpoint, num_fake, num_real, batch_size, seed):
    run_dir = os.path.join(PROJECT_ROOT, "experiments", run_folder)
    if not os.path.isdir(run_dir):
        print(f"[skip] {run_folder}: not a directory")
        return None
    ckpt_path = os.path.join(run_dir, checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"[skip] {run_folder}: missing {checkpoint}")
        return None

    model, cfg = build_model_from_run(run_dir)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # 兼容旧 checkpoint：
    #   - shape 完全一致 -> 直接加载
    #   - scalar gate [1] -> 通道级 gate [D]：广播为 D 维向量，保留训练得到的有效门控强度
    #   - 其它 shape 不匹配 -> 跳过该 key，保留模型初始化
    own_state = model.state_dict()
    filtered = {}
    dropped = []
    broadcasted = []
    for k, v in state.items():
        if k not in own_state:
            continue
        target_shape = own_state[k].shape
        if target_shape == v.shape:
            filtered[k] = v
        elif (
            tuple(v.shape) == (1,)
            and len(target_shape) == 1
            and target_shape[0] > 1
            and ".gate" in k
        ):
            filtered[k] = v.expand(target_shape[0]).clone().to(dtype=own_state[k].dtype)
            broadcasted.append((k, tuple(v.shape), tuple(target_shape)))
        else:
            dropped.append((k, tuple(v.shape), tuple(target_shape)))
    if broadcasted:
        warnings.warn(
            f"{run_folder}: broadcasted {len(broadcasted)} scalar-gate -> channel-gate (e.g. {broadcasted[:2]})"
        )
    if dropped:
        warnings.warn(
            f"{run_folder}: skipped {len(dropped)} shape-mismatch params (e.g. {dropped[:2]})"
        )
    missing = [k for k in own_state.keys() if k not in filtered]
    if missing:
        warnings.warn(
            f"{run_folder}: {len(missing)} params not in checkpoint (will keep init), e.g. {missing[:3]}"
        )
    model.load_state_dict(filtered, strict=False)
    model.eval()

    _m = cfg.get("model", {})
    cond_mode = str(_m.get("cond_mode", "attr")).strip().lower()
    latent_dim = int(_m.get("latent_dim", 128))
    clip_text_dim = int(_m.get("clip_text_dim", 512))
    img_size = int(cfg["train"]["img_size"])

    rng = torch.Generator(device="cpu").manual_seed(seed)
    py_rng = random.Random(seed)

    clip_seq_pool = None
    if cond_mode.startswith("clip_"):
        clip_cache_pt = _m.get("clip_cache_pt")
        per_image = load_clip_cache(clip_cache_pt)
        names = list(per_image.keys())
        if not names:
            raise RuntimeError(f"{run_folder}: CLIP 缓存为空")
        clip_seq_pool = (per_image, names)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid_resize = transforms.Resize((299, 299), antialias=True)

    real_names_all = _list_images(DATA_ROOT)
    if not real_names_all:
        raise RuntimeError("CelebA 训练目录为空")
    real_names = real_names_all[: min(num_real, len(real_names_all))]
    real_loader = DataLoader(
        RealImageDataset(DATA_ROOT, real_names, img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs in tqdm(real_loader, desc=f"{run_folder} [real]"):
            imgs = imgs.to(DEVICE)
            imgs = fid_resize(imgs)
            fid.update(imgs.clamp(0, 1), real=True)

        gen_count = 0
        pbar = tqdm(total=num_fake, desc=f"{run_folder} [fake]")
        while gen_count < num_fake:
            b = min(batch_size, num_fake - gen_count)
            z = torch.randn(b, latent_dim, generator=rng).to(DEVICE)
            cond = None
            if cond_mode.startswith("clip_"):
                per_image, names = clip_seq_pool
                seqs = []
                for _ in range(b):
                    n = py_rng.choice(names)
                    s = per_image[n].to(dtype=torch.float32)
                    seqs.append(s)
                # 统一序列长度
                target_len = max(s.shape[0] for s in seqs)
                padded = []
                for s in seqs:
                    if s.shape[0] < target_len:
                        pad = torch.zeros(target_len - s.shape[0], s.shape[1], dtype=torch.float32)
                        s = torch.cat([s, pad], dim=0)
                    padded.append(s)
                cond = torch.stack(padded, dim=0).to(DEVICE)
                if cond.shape[-1] != clip_text_dim:
                    raise RuntimeError(
                        f"{run_folder}: cache 维度 {cond.shape[-1]} 与 clip_text_dim {clip_text_dim} 不一致"
                    )
            recon = model.decode(z, cond)  # [-1, 1]
            recon = (recon * 0.5 + 0.5).clamp(0, 1)
            recon = fid_resize(recon)
            fid.update(recon, real=False)
            gen_count += b
            pbar.update(b)
        pbar.close()

    fid_val = fid.compute().item()
    dt = time.perf_counter() - t0
    print(f"[{run_folder}] FID={fid_val:.4f}  (num_real={len(real_names)}, num_fake={num_fake}, time={dt:.1f}s)")

    return {
        "Run Folder": run_folder,
        "Config Name": cfg["experiment"]["name"],
        "cond_mode": cond_mode,
        "Checkpoint": checkpoint,
        "num_real": len(real_names),
        "num_fake": num_fake,
        "img_size": img_size,
        "FID (↓)": fid_val,
        "Time (s)": round(dt, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="FID evaluation for Mamba-CVAE runs.")
    parser.add_argument("--exp-dirs", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str, default="model_epoch_20.pth")
    parser.add_argument("--num-fake", type=int, default=10000)
    parser.add_argument("--num-real", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    rows = []
    for d in args.exp_dirs:
        row = evaluate_one_run(
            d,
            checkpoint=args.checkpoint,
            num_fake=args.num_fake,
            num_real=args.num_real,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("没有评估到任何 run。")

    out_path = args.out_csv or os.path.join(
        PROJECT_ROOT, "experiments", f"fid_compare_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n========== FID Summary ==========")
    for r in rows:
        print(r)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
