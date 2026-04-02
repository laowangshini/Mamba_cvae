"""
重建任务定量对比：读取 experiments/ 下各次运行目录中的 run_config.yaml + 权重，计算 PSNR / SSIM / LPIPS / Params / MACs。

识别某次运行归属：
  - 每个运行目录内：run_config.yaml（训练配置快照）、manifest.json（人类可读的实验身份证）
  - 全局：experiments/experiment_registry.csv（按时间追加的索引表）

用法示例：
  python evaluate.py --auto-latest
  python evaluate.py --exp-dirs exp_baseline_cnn_20260329_032232 exp_a_mamba_1d_20260329_032233 exp_b_mamba_ss2d_20260329_032232
"""
import argparse
import csv
import os
import re
import time
import warnings

import pandas as pd
import torch
import yaml
from PIL import Image
import numpy as np
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.cvae import MambaCVAE

# ================= 配置区 =================
PROJECT_ROOT = "/root/autodl-tmp/Mamba-CVAE"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "celeba", "img_align_celeba")
ATTR_CSV = os.path.join(PROJECT_ROOT, "data", "celeba", "list_attr_celeba.csv")

DEFAULT_TRAIN_LIMIT = 50000
NUM_TEST = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CANONICAL_PREFIXES = ("exp_baseline_cnn", "exp_a_mamba_1d", "exp_b_mamba_ss2d")
# ========================================

DEFAULT_CLIP_LOCAL_PATH = "/root/autodl-tmp/CLIP"


def _list_images(img_dir):
    names = sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    return names


def load_celeba_attrs(attr_csv_path, cond_dim=40):
    attr_map = {}
    with open(attr_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 1 + cond_dim:
                continue
            name = row[0].strip()
            vals = [float(row[j]) for j in range(1, 1 + cond_dim)]
            attr_map[name] = torch.tensor(vals, dtype=torch.float32)
    return attr_map


def load_attr_names(attr_csv_path, cond_dim=40):
    with open(attr_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header[1 : 1 + cond_dim]]


def attrs_to_prompts(attr_tensor, attr_names):
    # attr_tensor: [B, cond_dim] with values roughly in {-1,1} or {0,1}
    out = []
    for row in attr_tensor:
        active = []
        for i, v in enumerate(row.tolist()):
            if v > 0:
                active.append(attr_names[i].replace("_", " ").lower())
        if not active:
            out.append("a face portrait")
        else:
            out.append("a face with " + ", ".join(active))
    return out


def compute_clip_score_batch(clip_model, clip_processor, images_01, prompts, device):
    # images_01: [B,3,H,W] in [0,1]
    imgs_np = (images_01.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(
        np.uint8
    )
    pil_images = [Image.fromarray(arr) for arr in imgs_np]
    inputs = clip_processor(
        text=prompts, images=pil_images, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = clip_model(**inputs)
    img_emb = F.normalize(outputs.image_embeds, p=2, dim=-1)
    txt_emb = F.normalize(outputs.text_embeds, p=2, dim=-1)
    cosine = (img_emb * txt_emb).sum(dim=-1)
    return (cosine * 100.0)


class CelebAHeldOutDataset(Dataset):
    """
    与 train.py 中「前 DEFAULT_TRAIN_LIMIT 张用于训练」对齐：
    优先使用排序文件名中索引在 [DEFAULT_TRAIN_LIMIT, DEFAULT_TRAIN_LIMIT+NUM_TEST) 的图像；
    若数量不足（小数据集），退化为取末尾若干张并在控制台提示可能与训练重叠。

    Phase3：cond_mode=clip_* 时返回 CLIP 文本序列 [L, C]（与 train 同源 clip_map）。
    """

    def __init__(
        self,
        img_dir,
        transform=None,
        num_train=DEFAULT_TRAIN_LIMIT,
        num_test=NUM_TEST,
        attr_map=None,
        cond_dim=40,
        cond_mode="attr",
        clip_map=None,
        clip_text_dim=768,
        clip_default_seq_len=77,
    ):
        self.img_dir = img_dir
        all_names = _list_images(img_dir)
        n = len(all_names)
        if n > num_train:
            self.img_names = all_names[num_train : min(num_train + num_test, n)]
            self.split_note = f"held-out indices [{num_train}, {num_train + len(self.img_names)})"
        else:
            k = min(num_test, max(1, n // 5))
            self.img_names = all_names[-k:]
            self.split_note = (
                f"fallback: last {k} images (dataset size {n} <= train limit {num_train}; "
                "may overlap with training — 仅适合内部对比，正式论文请换大数据或固定划分文件)"
            )
        self.transform = transform
        self.attr_map = attr_map
        self.cond_dim = cond_dim
        self.cond_mode = cond_mode
        self.clip_map = clip_map
        self.clip_text_dim = clip_text_dim
        self.clip_default_seq_len = clip_default_seq_len
        self.use_cond = (cond_mode == "attr" and attr_map is not None and cond_dim > 0) or (
            cond_mode.startswith("clip_") and clip_map is not None
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.cond_mode.startswith("clip_") and self.clip_map is not None:
            seq = self.clip_map.get(name)
            if seq is None:
                seq = torch.zeros(
                    self.clip_default_seq_len, self.clip_text_dim, dtype=torch.float32
                )
            else:
                seq = seq.to(dtype=torch.float32)
            return image, seq
        if self.use_cond:
            cond = self.attr_map.get(name)
            if cond is None:
                cond = torch.zeros(self.cond_dim, dtype=torch.float32)
            return image, cond
        return image


def discover_latest_run_dirs(exp_root, prefixes):
    """
    对每个前缀，选取 experiments 下目录名形如 {prefix}_YYYYMMDD_HHMMSS 中时间戳字典序最大的一个。
    """
    chosen = {}
    for prefix in prefixes:
        pat = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}$")
        candidates = []
        for d in os.listdir(exp_root):
            if pat.match(d) and os.path.isdir(os.path.join(exp_root, d)):
                candidates.append(d)
        if not candidates:
            chosen[prefix] = None
        else:
            candidates.sort()
            chosen[prefix] = candidates[-1]
    return chosen


def evaluate_model(exp_name_dir, checkpoint_name="model_latest.pth", max_batches=None):
    exp_path = os.path.join(PROJECT_ROOT, "experiments", exp_name_dir)
    config_path = os.path.join(exp_path, "run_config.yaml")
    weight_path = os.path.join(exp_path, checkpoint_name)

    if not os.path.isdir(exp_path):
        print(f"Skipping {exp_name_dir}: not a directory under experiments/.")
        return None
    if not os.path.exists(config_path) or not os.path.exists(weight_path):
        print(f"Skipping {exp_name_dir}: missing run_config.yaml or {checkpoint_name}.")
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    block_type = cfg["model"]["block_type"]
    latent_dim = cfg["model"]["latent_dim"]
    model_name = cfg["experiment"]["name"]
    _m = cfg.get("model", {})
    cond_dim = int(_m.get("cond_dim", 0))
    cond_embed_dim = int(_m.get("cond_embed_dim", 256))
    cond_mode = str(_m.get("cond_mode", "attr")).strip().lower()
    clip_text_dim = int(_m.get("clip_text_dim", 768))
    clip_cache_pt = _m.get("clip_cache_pt")
    mapper_bidirectional = bool(_m.get("mapper_bidirectional", True))
    clip_default_seq_len = int(_m.get("clip_seq_len", 77))
    attn_heads = int(_m.get("attn_heads", 4))
    bottleneck_inject_stages = int(_m.get("bottleneck_inject_stages", 1))
    # 与 train.py 一致：从本次运行的 run_config.yaml 读取，避免高分辨率训练后评估仍 Resize 成 64 导致崩溃或 FLOPs 统计错误
    img_size = cfg["train"]["img_size"]
    batch_size = cfg["train"]["batch_size"]

    model_cond_dim = 0 if cond_mode.startswith("clip_") else cond_dim
    model = MambaCVAE(
        latent_dim=latent_dim,
        block_type=block_type,
        cond_dim=model_cond_dim,
        cond_embed_dim=cond_embed_dim,
        cond_mode=cond_mode,
        clip_text_dim=clip_text_dim,
        mapper_bidirectional=mapper_bidirectional,
        attn_heads=attn_heads,
        bottleneck_inject_stages=bottleneck_inject_stages,
    ).to(DEVICE)
    state = torch.load(weight_path, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # 兼容旧 checkpoint：例如 gate 从标量升级为通道向量时形状不匹配
        warnings.warn(f"Non-strict load_state_dict due to mismatch: {e}")
        model.load_state_dict(state, strict=False)
    model.eval()

    try:
        if cond_mode.startswith("clip_"):

            class _FlopsWrapperClip(torch.nn.Module):
                def __init__(self, inner, seq_len, ctd):
                    super().__init__()
                    self.inner = inner
                    self.seq_len = seq_len
                    self.ctd = ctd

                def forward(self, x):
                    b = x.shape[0]
                    c = torch.zeros(b, self.seq_len, self.ctd, device=x.device, dtype=x.dtype)
                    recon, _, _ = self.inner(x, c)
                    return recon

            _wrap = _FlopsWrapperClip(
                model, clip_default_seq_len, clip_text_dim
            ).to(DEVICE)
            macs, params = get_model_complexity_info(
                _wrap, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False
            )
        elif cond_dim > 0:

            class _FlopsWrapper(torch.nn.Module):
                def __init__(self, inner, cd):
                    super().__init__()
                    self.inner = inner
                    self.cd = cd

                def forward(self, x):
                    b = x.shape[0]
                    c = torch.zeros(b, self.cd, device=x.device, dtype=x.dtype)
                    recon, _, _ = self.inner(x, c)
                    return recon

            _m = _FlopsWrapper(model, cond_dim).to(DEVICE)
            macs, params = get_model_complexity_info(
                _m, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False
            )
        else:
            macs, params = get_model_complexity_info(
                model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False
            )
    except Exception as e:
        warnings.warn(f"ptflops failed ({e}); Params/MACs set to nan.")
        macs, params = float("nan"), float("nan")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(DEVICE)

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    attr_map = None
    clip_map = None
    prompt_map = None
    attr_names = None
    clip_model = None
    clip_processor = None
    if cond_mode.startswith("clip_"):
        if not clip_cache_pt:
            print(f"Skipping {exp_name_dir}: cond_mode=clip_* 但未配置 clip_cache_pt")
            return None
        cpath = (
            clip_cache_pt
            if os.path.isabs(clip_cache_pt)
            else os.path.join(PROJECT_ROOT, clip_cache_pt)
        )
        if not os.path.isfile(cpath):
            print(f"Skipping {exp_name_dir}: 未找到 CLIP 缓存 {cpath}")
            return None
        try:
            blob = torch.load(cpath, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(cpath, map_location="cpu")
        if isinstance(blob, dict) and "per_image" in blob:
            clip_map = blob["per_image"]
            prompt_map = blob.get("prompt_per_image")
        else:
            clip_map = blob
    elif cond_dim > 0:
        if not os.path.isfile(ATTR_CSV):
            print(f"Skipping {exp_name_dir}: cond_dim>0 但未找到 {ATTR_CSV}")
            return None
        attr_map = load_celeba_attrs(ATTR_CSV, cond_dim=cond_dim)
        attr_names = load_attr_names(ATTR_CSV, cond_dim=cond_dim)

    dataset = CelebAHeldOutDataset(
        DATA_ROOT,
        transform=transform,
        attr_map=attr_map,
        cond_dim=cond_dim,
        cond_mode=cond_mode,
        clip_map=clip_map,
        clip_text_dim=clip_text_dim,
        clip_default_seq_len=clip_default_seq_len,
    )
    print(f"[{exp_name_dir}] Test split: {dataset.split_note} (n={len(dataset)})")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    total_clip = 0.0
    total_n = 0
    # 准备 CLIP 评估器（仅当可获取 prompt 时）
    if cond_mode.startswith("clip_") or (cond_dim > 0):
        try:
            from transformers import CLIPModel, CLIPProcessor

            clip_path = DEFAULT_CLIP_LOCAL_PATH
            clip_model = CLIPModel.from_pretrained(
                clip_path, local_files_only=True, use_safetensors=True
            ).to(DEVICE)
            clip_model.eval()
            clip_processor = CLIPProcessor.from_pretrained(
                clip_path, local_files_only=True
            )
        except Exception as e:
            warnings.warn(f"CLIP Score disabled (failed to load local CLIP at {DEFAULT_CLIP_LOCAL_PATH}: {e})")
            clip_model = None
            clip_processor = None

    with torch.no_grad():
        for bi, batch in enumerate(
            tqdm(dataloader, desc=f"Evaluating {model_name} ({exp_name_dir})")
        ):
            if max_batches is not None and bi >= max_batches:
                break
            if dataset.use_cond:
                images, cond = batch
                images = images.to(DEVICE)
                cond = cond.to(DEVICE)
                recon_images, _, _ = model(images, cond)
            else:
                images = batch.to(DEVICE)
                recon_images, _, _ = model(images)

            img_01 = (images * 0.5 + 0.5).clamp(0, 1)
            recon_01 = (recon_images * 0.5 + 0.5).clamp(0, 1)

            psnr_metric.update(recon_01, img_01)
            ssim_metric.update(recon_01, img_01)
            # LPIPS(net_type=vgg, normalize=True) 要求 [0,1] 与 NCHW
            lpips_metric.update(recon_01, img_01)

            if clip_model is not None and clip_processor is not None:
                prompts = None
                if cond_mode.startswith("clip_"):
                    if prompt_map is not None:
                        # 依赖 dataset.img_names 顺序：dataloader 不 shuffle
                        start = bi * batch_size
                        end = min(start + images.shape[0], len(dataset.img_names))
                        names = dataset.img_names[start:end]
                        prompts = [prompt_map.get(n, "a face portrait") for n in names]
                elif cond_dim > 0 and attr_names is not None and isinstance(cond, torch.Tensor):
                    prompts = attrs_to_prompts(cond, attr_names)

                if prompts is not None:
                    scores = compute_clip_score_batch(
                        clip_model, clip_processor, recon_01, prompts, DEVICE
                    )
                    total_clip += scores.sum().item()
                    total_n += scores.numel()

    psnr_val = psnr_metric.compute().item()
    ssim_val = ssim_metric.compute().item()
    lpips_val = lpips_metric.compute().item()
    clip_score = (total_clip / total_n) if total_n > 0 else float("nan")

    throughput_ips = None
    if DEVICE == "cuda" and len(dataset) > 0:
        # 固定小批量测速（与 1.txt 中 Throughput 描述一致）
        dummy = torch.randn(8, 3, img_size, img_size, device=DEVICE)
        if cond_mode.startswith("clip_"):
            dummy_cond = torch.randn(
                8, clip_default_seq_len, clip_text_dim, device=DEVICE
            )
        elif cond_dim > 0:
            dummy_cond = torch.randn(8, cond_dim, device=DEVICE)
        else:
            dummy_cond = None
        for _ in range(20):
            with torch.no_grad():
                if dummy_cond is not None:
                    _ = model(dummy, dummy_cond)
                else:
                    _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nrep = 100
        with torch.no_grad():
            for _ in range(nrep):
                if dummy_cond is not None:
                    _ = model(dummy, dummy_cond)
                else:
                    _ = model(dummy)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        throughput_ips = (8 * nrep) / dt

    manifest_path = os.path.join(exp_path, "manifest.json")
    has_manifest = os.path.isfile(manifest_path)

    return {
        "Run Folder": exp_name_dir,
        "Config Name": model_name,
        "Block Type": block_type,
        "cond_mode": cond_mode,
        "cond_dim": cond_dim,
        "img_size": img_size,
        "batch_size": batch_size,
        "Checkpoint": checkpoint_name,
        "Has manifest.json": has_manifest,
        "Params (M)": params / 1e6 if params == params else float("nan"),
        "MACs (G)": macs / 1e9 if macs == macs else float("nan"),
        "PSNR (↑)": psnr_val,
        "SSIM (↑)": ssim_val,
        "LPIPS (↓)": lpips_val,
        "CLIP Score (↑)": clip_score,
        "Throughput (img/s) @bs8": throughput_ips if throughput_ips is not None else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Mamba-CVAE experiment runs (quantitative)")
    parser.add_argument(
        "--exp-dirs",
        nargs="*",
        default=None,
        help="experiments/ 下的子目录名（可多个），例如 exp_b_mamba_ss2d_20260329_032232",
    )
    parser.add_argument(
        "--auto-latest",
        action="store_true",
        help=f"自动为 {CANONICAL_PREFIXES} 各选时间戳最新的一次运行目录",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_latest.pth",
        help="各运行目录内的权重文件名，如 model_latest.pth 或 model_epoch_20.pth",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="仅调试用：最多跑多少个 batch 后停止",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="汇总表保存路径（默认写到 experiments/comparison_时间戳.csv）",
    )
    args = parser.parse_args()

    exp_root = os.path.join(PROJECT_ROOT, "experiments")
    if args.auto_latest:
        picked = discover_latest_run_dirs(exp_root, CANONICAL_PREFIXES)
        exp_dirs = []
        for p in CANONICAL_PREFIXES:
            d = picked.get(p)
            if d is None:
                print(f"[auto-latest] 未找到前缀 {p}_YYYYMMDD_HHMMSS 的运行目录，跳过。")
            else:
                exp_dirs.append(d)
                print(f"[auto-latest] {p} -> {d}")
        if not exp_dirs:
            raise SystemExit("没有可用的运行目录，请先训练或改用 --exp-dirs 显式指定。")
    elif args.exp_dirs:
        exp_dirs = list(args.exp_dirs)
    else:
        raise SystemExit("请指定 --auto-latest 或使用 --exp-dirs 传入至少一个目录名。")

    all_rows = []
    for d in exp_dirs:
        row = evaluate_model(d, checkpoint_name=args.checkpoint, max_batches=args.max_batches)
        if row is not None:
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("没有成功评估任何运行。")

    df = pd.DataFrame(all_rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out_csv or os.path.join(exp_root, f"comparison_{ts}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n========== Summary ==========")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")
    print(
        "\n如何辨认 experiments/ 下「这是哪次模型」：\n"
        "  1) 打开该次目录下的 manifest.json（实验身份证）或 run_config.yaml；\n"
        "  2) 查看 experiments/experiment_registry.csv 按时间查找 run_folder 与 absolute_path；\n"
        "  3) 目录名本身含实验名 + 时间戳：{experiment_name}_{YYYYMMDD}_{HHMMSS}。"
    )


if __name__ == "__main__":
    main()
