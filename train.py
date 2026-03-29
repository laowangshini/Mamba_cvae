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
    """返回图像；若提供 attr_map 则同时返回 (img, cond) 的 40 维属性。"""

    def __init__(self, img_dir, transform=None, limit=None, attr_map=None, cond_dim=40):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        if limit:
            self.img_names = self.img_names[:limit]
        self.transform = transform
        self.attr_map = attr_map
        self.cond_dim = cond_dim
        self.use_cond = attr_map is not None and cond_dim > 0

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

        if self.use_cond:
            cond = self.attr_map.get(name)
            if cond is None:
                cond = torch.zeros(self.cond_dim, dtype=torch.float32)
            return image, cond
        return image


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD


def main():
    logger.info("="*50)
    logger.info(f"Starting Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Block Type: {BLOCK_TYPE}")
    logger.info(f"cond_dim: {COND_DIM}, cond_embed_dim: {COND_EMBED_DIM}")
    logger.info(f"Experiment Directory: {EXP_DIR}")
    logger.info(f"Manifest: {os.path.join(EXP_DIR, 'manifest.json')} (本目录归属说明)")
    logger.info(f"Registry: {os.path.join(PROJECT_ROOT, 'experiments', 'experiment_registry.csv')} (全局索引)")
    logger.info("="*50)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    attr_map = None
    if COND_DIM > 0:
        if not os.path.isfile(ATTR_CSV):
            raise FileNotFoundError(f"需要属性文件: {ATTR_CSV}")
        attr_map = load_celeba_attrs(ATTR_CSV, cond_dim=COND_DIM)
        logger.info(f"Loaded {len(attr_map)} attribute rows from list_attr_celeba.csv")

    dataset = CelebAImageDataset(
        DATA_ROOT,
        transform=transform,
        limit=50000,
        attr_map=attr_map,
        cond_dim=COND_DIM,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    logger.info(f"Dataset loaded with {len(dataset)} images.")

    model = MambaCVAE(
        latent_dim=LATENT_DIM,
        block_type=BLOCK_TYPE,
        cond_dim=COND_DIM,
        cond_embed_dim=COND_EMBED_DIM,
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    logger.info("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        last_cond = None

        for batch in pbar:
            if COND_DIM > 0:
                images, cond = batch
                images = images.to(DEVICE)
                cond = cond.to(DEVICE)
                last_cond = cond
            else:
                images = batch.to(DEVICE)
                cond = None

            optimizer.zero_grad()
            recon_images, mu, logvar = model(images, cond)
            loss = loss_function(recon_images, images, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item() / len(images)})

        avg_loss = total_loss / len(dataset)
        logger.info(f"Epoch[{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(EXP_DIR, "model_latest.pth"))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(EXP_DIR, f"model_epoch_{epoch+1}.pth"))
            save_reconstruction(
                model, images[:8], last_cond[:8] if last_cond is not None else None, epoch+1, EXP_DIR
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
