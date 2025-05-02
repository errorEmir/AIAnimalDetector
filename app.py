import gradio as gr

from predict import predict_image, load_model_and_classes  # Güncellenmiş yükleme fonksiyonunu import et

# --- Gradio Arayüzü ---

print("Setting up Gradio interface...")

# Uygulama başlarken modeli ve sınıf isimlerini yüklemeyi dene
# Bu, ilk istek gelmeden önce yüklemeyi sağlar, yanıt süresini iyileştirir.
initial_load_success = load_model_and_classes()

# Arayüz başlığı ve açıklaması
title = "Hayvan Görüntü Sınıflandırıcı (Animals-10)"
description = """
Bu uygulama, Animals-10 veri seti ile eğitilmiş bir ResNet-18 modeli kullanarak
görüntüleri 10 hayvan sınıfından birine (kelebek, kedi, tavuk, inek, köpek, fil, at, koyun, örümcek, sincap) ayırır.
Bilgisayarınızdan bir hayvan resmi yükleyin ve 'Tahmin Et' butonuna tıklayın.
Model CUDA (GPU) desteği ile çalışacak şekilde ayarlanmıştır (eğer mevcutsa).
"""
article = "<p style='text-align: center'>Demo for AI Image Classifier Project | Model: ResNet-18 | Dataset: Animals-10</p>"

# Gradio arayüzünü oluştur
# predict_image fonksiyonu zaten bir dictionary döndürüyor (olasılıklar veya hata)
# Gradio'nun Label bileşeni bu dictionary'yi otomatik olarak işleyebilir.
iface = gr.Interface(
    fn=predict_image,        # Tahmin fonksiyonu (predict.py'den)
    inputs=gr.Image(type="pil", label="Hayvan Resmi Yükle"), # Girdi: PIL formatında resim
    outputs=gr.Label(num_top_classes=3, label="Tahmin Sonuçları"), # Çıktı: En yüksek 3 sınıfı gösteren etiket
    title=title,
    description=description,
    article=article,
    allow_flagging="never"
)

print("Gradio interface setup complete.")

# Arayüzü başlat
if __name__ == "__main__":
    print("Launching Gradio app...")
    if not initial_load_success:
         print("\nUYARI: Model veya sınıf isimleri başlangıçta yüklenemedi.")
         print("Lütfen 'models/animal_classifier_resnet18.pth' ve 'models/class_names.json' dosyalarının var olduğundan emin olun.")
         print("Eğitim betiğini (train.py) çalıştırmanız gerekebilir.\n")

    # share=True dış ağdan erişim sağlar (dikkatli kullanın)
    # server_name="0.0.0.0" yerel ağdaki diğer cihazlardan erişim sağlar
    # Sadece kendi bilgisayarınızda çalıştırmak için server_name="127.0.0.1" veya varsayılanı kullanın
    iface.launch(server_name="127.0.0.1") # veya iface.launch()