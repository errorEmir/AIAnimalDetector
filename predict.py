import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from PIL import Image
import os
# utils aynı dizinde olduğu için import doğru
from utils import val_predict_transform, get_device, load_class_names

# --- Ayarlar (Yollar Güncellendi) ---
MODEL_PATH = 'models/animal_classifier_resnet18.pth'
CLASS_NAMES_PATH = 'models/class_names.json'
NUM_CLASSES = 10 # Eğitimdeki ile aynı olmalı

# --- Global Değişkenler ---
device = get_device()
model = None
class_names = None
model_loaded_successfully = False # Modelin başarılı yüklenip yüklenmediğini takip et

def load_model_and_classes():
    """Modeli ve sınıf isimlerini yükler."""
    global model, class_names, model_loaded_successfully, device

    if model_loaded_successfully: # Zaten başarılı yüklendiyse tekrar deneme
        return True

    print("Attempting to load model and class names...")
    # Sınıf isimlerini yükle
    loaded_names = load_class_names(CLASS_NAMES_PATH)
    if loaded_names is None:
        model_loaded_successfully = False
        return False # Sınıf isimleri olmadan devam edilemez

    # Sınıf isimleri yüklendiyse global değişkene ata
    class_names = loaded_names

    # Modeli tanımla
    try:
        # weights=None kullanıyoruz çünkü kendi ağırlıklarımızı yükleyeceğiz
        loaded_model = models.resnet18(weights=None)
        num_ftrs = loaded_model.fc.in_features
        # NUM_CLASSES'ın doğru olduğundan emin ol (class_names ile tutarlı)
        if len(class_names) != NUM_CLASSES:
             print(f"Warning: Number of classes in {CLASS_NAMES_PATH} ({len(class_names)}) doesn't match NUM_CLASSES ({NUM_CLASSES}). Using {len(class_names)} classes.")
             actual_num_classes = len(class_names)
        else:
             actual_num_classes = NUM_CLASSES

        loaded_model.fc = nn.Linear(num_ftrs, actual_num_classes)

        # Kaydedilmiş ağırlıkları yükle
        if not os.path.exists(MODEL_PATH):
             print(f"Error: Model file not found at {MODEL_PATH}")
             print("Please ensure you have run the training script first (train.py).")
             model_loaded_successfully = False
             return False

        # map_location=device: modeli CPU'da çalıştırırken GPU'da eğitilmiş modeli yüklemek için
        loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        loaded_model.eval() # Değerlendirme moduna al
        model = loaded_model.to(device) # Cihaza taşı
        model_loaded_successfully = True # Başarılı yüklendi
        print("Model and class names loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        model = None # Hata durumunda modeli None yap
        model_loaded_successfully = False
        return False


def predict_image(image: Image.Image):
    """
    Verilen PIL Image nesnesini sınıflandırır ve sınıf olasılıklarını döndürür.
    Gradio arayüzü ile kullanılmak üzere tasarlanmıştır.
    """
    global model, class_names, model_loaded_successfully

    # Modelin ve sınıfların yüklenip yüklenmediğini kontrol et, gerekirse yükle
    if not model_loaded_successfully:
        if not load_model_and_classes():
             # Yükleme başarısız olduysa Gradio'ya anlamlı bir hata döndür
             return {"Error": "Model veya sınıf isimleri yüklenemedi. Lütfen train.py betiğini çalıştırdığınızdan ve 'models' klasörünün doğru yerde olduğundan emin olun."}

    if model is None or class_names is None:
         return {"Error": "Model veya sınıf isimleri mevcut değil."}

    try:
        # Görüntüyü ön işle
        img_rgb = image.convert('RGB')
        # val_predict_transform utils'dan geliyor
        input_tensor = val_predict_transform(img_rgb).unsqueeze(0).to(device)

        # Tahmin yap
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]

        # Sonuçları formatla (Sınıf Adı: Olasılık)
        # class_names listesinin index'leri ile probabilities tensor'ünün index'leri eşleşmeli
        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        # print(f"Prediction confidences: {confidences}") # Debugging için

        return confidences

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Kullanıcıya daha genel bir hata mesajı gösterilebilir
        return {"Error": f"Tahmin sırasında bir hata oluştu."}

# Komut satırından test etmek için (opsiyonel)
if __name__ == '__main__':
    # Modelin ve sınıfların yüklenmesini sağla
    if not load_model_and_classes():
        print("Exiting due to loading failure.")
    else:
        # Örnek bir resim yolu isteyin
        try:
            # raw-img klasöründen bir resim yolu girilmesini bekleyebiliriz
            # Örnek: raw-img/dog/xxx.jpg veya raw-img/cat/yyy.jpg
            image_path = input("Tahmin edilecek resmin yolunu girin (örn: raw-img/sinif_adi/resim.jpg): ")
            if not os.path.exists(image_path):
                print(f"Hata: '{image_path}' bulunamadı.")
            else:
                img = Image.open(image_path)
                predictions = predict_image(img) # predict_image zaten PIL Image bekliyor
                print("\nTahmin Sonuçları:")
                if "Error" not in predictions:
                    # Sonuçları olasılığa göre sırala
                    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
                    for class_name, prob in sorted_predictions:
                        print(f"  - {class_name}: {prob:.4f}")
                else:
                    print(predictions["Error"])
        except FileNotFoundError:
             print("Hata: Belirtilen dosya bulunamadı.")
        except Exception as e:
            print(f"Bir hata oluştu: {e}")