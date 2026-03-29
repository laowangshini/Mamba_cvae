import os
import torch
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# ================= 配置区域 =================
# 适配新的数据路径
PROJECT_ROOT = "/root/autodl-tmp/Mamba-CVAE"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "celeba")
IMG_DIR = os.path.join(DATA_ROOT, "img_align_celeba")
ATTR_FILE = os.path.join(DATA_ROOT, "list_attr_celeba.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "celeba_clip_features.pkl")

BATCH_SIZE = 64  # 根据显存调整
MAX_SAMPLES = 10000  # 限制处理数量，设为 None 则处理全部
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def generate_text_prompt(row):
    """
    根据 CelebA 属性生成文本描述
    row: pandas Series, 包含属性 (+1/-1)
    """
    # 基础描述
    gender = "man" if row['Male'] == 1 else "woman"
    desc = [f"A photo of a {gender}"]
    
    # 定义属性映射 (只选部分明显的特征)
    attr_map = {
        'Smiling': 'smiling',
        'Young': 'young',
        'Eyeglasses': 'wearing eyeglasses',
        'Wearing_Hat': 'wearing a hat',
        'Blond_Hair': 'with blond hair',
        'Black_Hair': 'with black hair',
        'Brown_Hair': 'with brown hair',
        'Gray_Hair': 'with gray hair',
        'Bald': 'who is bald',
        'Mustache': 'with a mustache',
        'No_Beard': 'clean-shaven',
        'Pale_Skin': 'with pale skin',
        'Bangs': 'with bangs',
        'Straight_Hair': 'with straight hair',
        'Wavy_Hair': 'with wavy hair',
        'Attractive': 'attractive'
    }
    
    features = []
    for col, text in attr_map.items():
        if row.get(col, -1) == 1:
            features.append(text)
            
    if features:
        desc.append("who is " + ", ".join(features))
        
    return " ".join(desc)

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None, limit=None):
        self.img_dir = img_dir
        # 读取 CSV
        print(f"Loading attributes from {attr_file}...")
        self.df = pd.read_csv(attr_file)
        
        if limit:
            self.df = self.df.iloc[:limit]
            
        self.img_names = self.df['image_id'].tolist()
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 加载图片
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个纯黑图片防止中断
            image = Image.new('RGB', (178, 218), (0, 0, 0))
            
        # 生成文本
        text = generate_text_prompt(row)
        
        return image, text

# 自定义 collate 函数，处理 PIL Image 和 text
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 1. 加载 CLIP（从本地路径）
    CLIP_MODEL_PATH = "/root/autodl-tmp/CLIP"
    print(f"Loading CLIP model from local path: {CLIP_MODEL_PATH}")
    
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
        print("Model loaded successfully from local directory!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # 2. 准备数据
    dataset = CelebADataset(IMG_DIR, ATTR_FILE, limit=MAX_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=4, collate_fn=custom_collate_fn)

    results = []

    print(f"Start extracting features for {len(dataset)} images...")
    
    model.eval()
    with torch.no_grad():
        for images, texts in tqdm(loader):
            # CLIP 预处理
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            # 提取图像特征 (Image Embedding) -> (B, 512)
            image_features = model.get_image_features(**inputs)
            
            # 提取文本特征 (Text Sequence) -> Mamba 需要序列信息
            text_outputs = model.text_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            # last_hidden_state: (B, Sequence_Length, Hidden_Size) -> (B, 77, 512)
            text_seq = text_outputs.last_hidden_state 
            
            # pooled_output: (B, 512)
            text_pooled = text_outputs.pooler_output

            # 转移到 CPU 并存储
            image_features_np = image_features.cpu().numpy()
            text_seq_np = text_seq.cpu().numpy()
            text_pooled_np = text_pooled.cpu().numpy()
            
            for i in range(len(images)):
                results.append({
                    "image_emb": image_features_np[i],
                    "text_seq": text_seq_np[i],
                    "text_emb": text_pooled_np[i],
                    "text_raw": texts[i]
                })

    # 3. 保存结果
    print(f"Saving {len(results)} items to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f)
    print("Done!")

if __name__ == "__main__":
    main()

