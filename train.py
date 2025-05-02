import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import models
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# utils aynı dizinde olduğu için import doğru
from utils import train_transform, val_predict_transform, save_class_names, get_device
# Multiprocessing için freeze_support ekleyin (Windows için önemli)
from multiprocessing import freeze_support # Bu satırı ekleyin

# --- Ayarlar ---
# Bu ayarlar global kalabilir veya main fonksiyonuna taşınabilir
DATA_DIR = 'raw-img'
MODEL_SAVE_PATH = 'models/animal_classifier_resnet18.pth'
CLASS_NAMES_PATH = 'models/class_names.json'
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# --- Fonksiyonlar (main dışında tanımlanabilir) ---
# get_device gibi yardımcı fonksiyonlar burada veya utils'de kalabilir
# Model tanımı gibi şeyler de burada olabilir, ancak çağrılması main içinde olmalı

def run_training():
    """Eğitim sürecini çalıştıran ana fonksiyon."""
    # --- Cihaz Seçimi ---
    device = get_device()

    # --- Veri Yükleme ve Hazırlama ---
    print("Loading dataset...")
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please ensure the 'raw-img' folder exists in the same directory as train.py and contains the class folders.")
        return # Hata durumunda fonksiyondan çık

    full_dataset = ImageFolder(DATA_DIR, transform=train_transform)

    class_names = full_dataset.classes
    if len(class_names) != NUM_CLASSES:
         print(f"Warning: Found {len(class_names)} classes in {DATA_DIR}, but expected {NUM_CLASSES}.")
    print(f"Classes found: {class_names}")
    save_class_names(class_names, CLASS_NAMES_PATH)

    num_data = len(full_dataset)
    num_val = int(VALIDATION_SPLIT * num_data)
    num_train = num_data - num_val
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    # Doğrulama seti için transformu ayarla
    # Dikkat: Bu orijinal dataset'in transformunu değiştirir. Alternatif olarak kopya oluşturulabilir.
    val_dataset.dataset.transform = val_predict_transform

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # DataLoader'ları oluştur (num_workers > 0 ise __main__ bloğu önemli)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("Data loaders created.")

    # --- Model Tanımı ---
    print("Loading pre-trained ResNet-18 model...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)
    print("Model loaded and moved to device.")

    # --- Kayıp Fonksiyonu ve Optimizatör ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- Eğitim Döngüsü ---
    print("Starting training...")
    best_val_accuracy = 0.0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        # Bu döngü DataLoader'ı kullanır ve worker'ları tetikler
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)
        epoch_train_acc = accuracy_score(train_labels, train_preds)

        # --- Doğrulama Döngüsü ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Duration: {epoch_duration:.2f}s")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}   | Val Acc: {epoch_val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

        scheduler.step()

        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"----> Best model saved to {MODEL_SAVE_PATH} with Validation Accuracy: {best_val_accuracy:.4f}")

    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    # --- Dokümantasyon için Metrikler ---
    print("\n--- Metrics for Documentation (Best Model based on Validation Accuracy) ---")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    # Son epoch'un P/R/F1 değerleri en iyi modele ait olmayabilir, dikkat!
    # print(f"Validation Precision (Macro, last epoch): {precision:.4f}")
    # print(f"Validation Recall (Macro, last epoch): {recall:.4f}")
    # print(f"Validation F1-Score (Macro, last epoch): {f1:.4f}")


# --- Ana Çalıştırma Bloğu ---
if __name__ == '__main__':
    # Windows'ta DataLoader worker'ları için freeze_support() çağrısı önemlidir.
    # Eğer programınızı .exe gibi dondurulmuş bir pakete çevirmeyecekseniz
    # teknik olarak zorunlu olmayabilir, ama eklemek genellikle güvenlidir.
    freeze_support() # Bu satırı ekleyin

    # Eğitimi başlat
    run_training()