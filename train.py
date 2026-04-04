import os
import csv
import json
import time
import logging
import argparse
import yaml
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.cvae import MambaCVAE

# ================= 动态配置加载 =================
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ================= 命令行参数 =================
parser = argparse.ArgumentParser(description="Train Mamba-CVAE")
parser.add_argument('--config', type=str, default='configs/exp_b_mamba_ss2d.yaml', help='Path to config file')
args = parser.parse_args()

cfg = load_config(args.config)

# ================= 从配置中读取变量 =================
PROJECT_ROOT = "/root/autodl-tmp/Mamba-CVAE"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "celeba", "img_align_celeba")
ATTR_CSV = os.path.join(PROJECT_ROOT, "data", "celeba", "list_attr_celeba.csv")

EXPERIMENT_NAME = cfg['experiment']['name']
BLOCK_TYPE = cfg['model']['block_type']
LATENT_DIM = cfg['model']['latent_dim']
BATCH_SIZE = cfg['train']['batch_size']
LR = float(cfg['train']['lr'])
EPOCHS = cfg['train']['epochs']
IMG_SIZE = cfg['train']['img_size']

_model = cfg.get('model', {})
COND_DIM = int(_model.get('cond_dim', 0))
COND_EMBED_DIM = int(_model.get('cond_embed_dim', 256))
COND_MODE = str(_model.get('cond_mode', 'attr')).strip().lower()
CLIP_TEXT_DIM = int(_model.get('clip_text_dim', 768))
CLIP_CACHE_PT = _model.get('clip_cache_pt')
MAPPER_BIDIRECTIONAL = bool(_model.get('mapper_bidirectional', True))
ATTN_HEADS = int(_model.get('attn_heads', 4))
BOTTLENECK_INJECT_STAGES = int(_model.get('bottleneck_inject_stages', 1))
KLD_WEIGHT = float(cfg['train'].get('kld_weight', 1.0))
LAMBDA_LPIPS = float(cfg['train'].get('lambda_lpips', 0.0))
COND_DROPOUT_PROB = float(cfg['train'].get('cond_dropout_prob', 0.0))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 实验目录创建与备份 ---
CURRENT_TIME = time.strftime("%Y%m%d_%H%M%S")
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", f"{EXPERIMENT_NAME}_{CURRENT_TIME}")
os.makedirs(EXP_DIR, exist_ok=True)

shutil.copy(args.config, os.path.join(EXP_DIR, "run_config.yaml"))

# 单次运行的“身份证”：与 run_config.yaml 并列，便于在 experiments/ 下快速识别归属
_manifest = {
    "run_folder": os.path.basename(EXP_DIR),
    "absolute_path": EXP_DIR,
    "experiment_name": EXPERIMENT_NAME,
    "block_type": BLOCK_TYPE,
    "latent_dim": LATENT_DIM,
    "cond_dim": COND_DIM,
    "cond_embed_dim": COND_EMBED_DIM,
    "cond_mode": COND_MODE,
    "clip_text_dim": CLIP_TEXT_DIM,
    "clip_cache_pt": (
        os.path.join(PROJECT_ROOT, CLIP_CACHE_PT)
        if CLIP_CACHE_PT and not os.path.isabs(CLIP_CACHE_PT)
        else CLIP_CACHE_PT
    ),
    "mapper_bidirectional": MAPPER_BIDIRECTIONAL,
    "attn_heads": ATTN_HEADS,
    "bottleneck_inject_stages": BOTTLENECK_INJECT_STAGES,
    "description": cfg["experiment"].get("description", ""),
    "train": dict(cfg["train"]),
    "source_config": os.path.abspath(
        args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    ),
    "started_at_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
    "timestamp_suffix": CURRENT_TIME,
}
with open(os.path.join(EXP_DIR, "manifest.json"), "w", encoding="utf-8") as _mf:
    json.dump(_manifest, _mf, indent=2, ensure_ascii=False)

_registry = os.path.join(PROJECT_ROOT, "experiments", "experiment_registry.csv")
_write_header = not os.path.exists(_registry) or os.path.getsize(_registry) == 0
with open(_registry, "a", newline="", encoding="utf-8") as _rf:
    _w = csv.writer(_rf)
    if _write_header:
        _w.writerow(
            ["created_at", "run_folder", "experiment_name", "block_type", "absolute_path"]
        )
    _w.writerow([CURRENT_TIME, os.path.basename(EXP_DIR), EXPERIMENT_NAME, BLOCK_TYPE, EXP_DIR])

# 设置日志系统
log_file = os.path.join(EXP_DIR, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


def load_celeba_attrs(attr_csv_path, cond_dim=40):
    """image_id -> cond 向量，值为 {-1,1} 转为 float。"""
    attr_map = {}
    with open(attr_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # 第一列为 image_id，其后 cond_dim 个属性
        for row in reader:
            if len(row) < 1 + cond_dim:
                continue
            name = row[0].strip()
            vals = []
            for j in range(1, 1 + cond_dim):
                vals.append(float(row[j]))
            attr_map[name] = torch.tensor(vals, dtype=torch.float32)
    return attr_map


class CelebAImageDataset(Dataset):
    """
    返回图像；
    Phase2：attr_map + cond_dim>0 时返回 (img, cond) 属性向量；
    Phase3：cond_mode=clip_* 且 clip_map 时返回 (img, text_seq)，text_seq 形状 [L, C]。
    """

    def __init__(
        self,
        img_dir,
        transform=None,
        limit=None,
        attr_map=None,
        cond_dim=40,
        cond_mode="attr",
        clip_map=None,
        clip_text_dim=768,
        clip_default_seq_len=77,
    ):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        if limit:
            self.img_names = self.img_names[:limit]
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
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception:
            image = torch.zeros(3, IMG_SIZE, IMG_SIZE)

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


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    L1 = F.l1_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return L1 + beta * KLD


def main():
    logger.info("="*50)
    logger.info(f"Starting Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Block Type: {BLOCK_TYPE}")
    logger.info(
        f"cond_mode: {COND_MODE}, cond_dim: {COND_DIM}, cond_embed_dim: {COND_EMBED_DIM}, "
        f"clip_text_dim: {CLIP_TEXT_DIM}"
    )
    logger.info(f"Experiment Directory: {EXP_DIR}")
    logger.info(f"Manifest: {os.path.join(EXP_DIR, 'manifest.json')} (本目录归属说明)")
    logger.info(f"Registry: {os.path.join(PROJECT_ROOT, 'experiments', 'experiment_registry.csv')} (全局索引)")
    logger.info("="*50)

    loss_fn_vgg = None
    if LAMBDA_LPIPS > 0:
        try:
            import lpips
        except Exception as e:
            raise ImportError(
                "需要安装 lpips 才能启用 perceptual loss：pip install lpips"
            ) from e
        logger.info(f"[INFO] 启用 LPIPS Loss (lambda_lpips={LAMBDA_LPIPS})")
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        for p in loss_fn_vgg.parameters():
            p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    attr_map = None
    clip_map = None
    model_cond_dim = COND_DIM

    if COND_MODE.startswith("clip_"):
        if not CLIP_CACHE_PT:
            raise FileNotFoundError(
                "Phase3 cond_mode=clip_* 需要配置 model.clip_cache_pt（CLIP 文本序列缓存 .pt）。"
            )
        cache_path = (
            CLIP_CACHE_PT
            if os.path.isabs(CLIP_CACHE_PT)
            else os.path.join(PROJECT_ROOT, CLIP_CACHE_PT)
        )
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"未找到 CLIP 缓存: {cache_path}")
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(cache_path, map_location="cpu")
        if isinstance(blob, dict) and "per_image" in blob:
            clip_map = blob["per_image"]
            fdim = blob.get("clip_text_dim")
            if fdim is not None and int(fdim) != CLIP_TEXT_DIM:
                logger.warning(
                    "缓存 clip_text_dim=%s 与配置 %s 不一致，请与预计算脚本/YAML 对齐。",
                    fdim,
                    CLIP_TEXT_DIM,
                )
        else:
            clip_map = blob
        logger.info("Loaded CLIP text-seq cache for %d filenames.", len(clip_map))
        model_cond_dim = 0
    elif COND_DIM > 0:
        if not os.path.isfile(ATTR_CSV):
            raise FileNotFoundError(f"需要属性文件: {ATTR_CSV}")
        attr_map = load_celeba_attrs(ATTR_CSV, cond_dim=COND_DIM)
        logger.info(f"Loaded {len(attr_map)} attribute rows from list_attr_celeba.csv")
    elif COND_MODE != "attr":
        raise ValueError(
            f"未知 cond_mode: {COND_MODE}（支持 attr | clip_pooled | clip_attention | clip_seq）"
        )

    dataset = CelebAImageDataset(
        DATA_ROOT,
        transform=transform,
        limit=50000,
        attr_map=attr_map,
        cond_dim=COND_DIM,
        cond_mode=COND_MODE,
        clip_map=clip_map,
        clip_text_dim=CLIP_TEXT_DIM,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    logger.info(f"Dataset loaded with {len(dataset)} images.")

    model = MambaCVAE(
        latent_dim=LATENT_DIM,
        block_type=BLOCK_TYPE,
        cond_dim=model_cond_dim,
        cond_embed_dim=COND_EMBED_DIM,
        cond_mode=COND_MODE,
        clip_text_dim=CLIP_TEXT_DIM,
        mapper_bidirectional=MAPPER_BIDIRECTIONAL,
        attn_heads=ATTN_HEADS,
        bottleneck_inject_stages=BOTTLENECK_INJECT_STAGES,
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    logger.info("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        last_cond_vis = None
        cond_dropped_rate = COND_DROPOUT_PROB if (COND_MODE.startswith("clip_") and COND_DROPOUT_PROB > 0) else 0.0

        for batch in pbar:
            if dataset.use_cond:
                images, cond = batch
                images = images.to(DEVICE)
                cond = cond.to(DEVICE)
                # 训练期 CFG dropout：以一定概率将条件置零，等价“空文本”训练
                # 注意：last_cond_vis 保留为未置空版本，用于可视化重建
                last_cond_vis = cond
                if cond_dropped_rate > 0:
                    b = cond.shape[0]
                    drop = (torch.rand(b, device=cond.device) < cond_dropped_rate)
                    keep = (~drop).float()
                    if cond.dim() == 3:
                        keep = keep[:, None, None]
                    elif cond.dim() == 2:
                        keep = keep[:, None]
                    else:
                        raise ValueError(f"Unexpected cond shape for dropout: {cond.shape}")
                    cond = cond * keep
            else:
                images = batch.to(DEVICE)
                cond = None

            optimizer.zero_grad()
            recon_images, mu, logvar = model(images, cond)
            base_loss = loss_function(recon_images, images, mu, logvar, beta=KLD_WEIGHT)
            loss = base_loss
            if loss_fn_vgg is not None:
                # recon_images / images: [-1,1] -> [0,1] for LPIPS
                recons_01 = (recon_images * 0.5 + 0.5).clamp(0, 1)
                imgs_01 = (images * 0.5 + 0.5).clamp(0, 1)
                lpips_loss = loss_fn_vgg(recons_01, imgs_01).mean()
                loss = base_loss + LAMBDA_LPIPS * lpips_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch[{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(EXP_DIR, "model_latest.pth"))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(EXP_DIR, f"model_epoch_{epoch+1}.pth"))
            save_reconstruction(
                model, images[:8], last_cond_vis[:8] if last_cond_vis is not None else None, epoch+1, EXP_DIR
            )

    logger.info("Training completed.")


def save_reconstruction(model, originals, cond_batch, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        if cond_batch is not None:
            recons, _, _ = model(originals, cond_batch)
        else:
            recons, _, _ = model(originals)

    originals = (originals * 0.5 + 0.5).cpu()
    recons = (recons * 0.5 + 0.5).cpu()

    n = len(originals)
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))
    for i in range(n):
        axes[0, i].imshow(originals[i].permute(1, 2, 0).clip(0,1))
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i].permute(1, 2, 0).clip(0,1))
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"recon_epoch_{epoch}.png"))
    plt.close()
    model.train()


if __name__ == "__main__":
    main()
