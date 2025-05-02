import torch
import torchvision.transforms as transforms
from PIL import Image
import json

# Ön işleme adımları (ImageNet istatistikleri genellikle iyi bir başlangıçtır)
# Eğitim için: Biraz daha fazla Augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),             # Görüntüyü 256x256'ya getir
    transforms.RandomResizedCrop(224),  # 224x224 rastgele kırp
    transforms.RandomHorizontalFlip(),  # Rastgele yatay çevir
    transforms.ToTensor(),              # Görüntüyü PyTorch tensor'üne çevir
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize et
])

# Doğrulama/Test/Tahmin için: Augmentation olmadan
val_predict_transform = transforms.Compose([
    transforms.Resize(256),             # Görüntüyü 256x256'ya getir
    transforms.CenterCrop(224),         # Ortadan 224x224 kırp
    transforms.ToTensor(),              # Görüntüyü PyTorch tensor'üne çevir
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize et
])

def preprocess_image(image_path):
    """Tek bir görüntüyü tahmin için ön işler."""
    try:
        img = Image.open(image_path).convert('RGB')
        return val_predict_transform(img).unsqueeze(0) # Batch boyutu ekle (1, C, H, W)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def save_class_names(class_names, file_path="models/class_names.json"):
    """Sınıf isimlerini JSON olarak kaydeder."""
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {file_path}")

def load_class_names(file_path="models/class_names.json"):
    """Kaydedilmiş sınıf isimlerini yükler."""
    try:
        with open(file_path, 'r') as f:
            class_names = json.load(f)
        print(f"Class names loaded from {file_path}")
        return class_names
    except FileNotFoundError:
        print(f"Error: Class names file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None

# Cihazı belirle (CUDA varsa GPU, yoksa CPU)
def get_device():
    """Kullanılabilir en iyi cihazı (GPU/CPU) döndürür."""
    if torch.cuda.is_available():
        print("CUDA (GPU) is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA (GPU) not available. Using CPU.")
        return torch.device("cpu")