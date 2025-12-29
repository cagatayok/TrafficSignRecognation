"""
GTSRB GERÇEK VERİ SETİ İLE MODEL EĞİTİMİ + TESPİT
Klasör yapısı: gtsrb/Training/00000/*.ppm
"""

import os

# --- AYAR: TensorFlow uyumluluk ayarı ---
# Bazı yeni sürümlerde hata vermemesi için eski Keras formatını kullanır.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import subprocess


def install_packages():
    """
    --- OTOMATİK KURULUM FONKSİYONU ---
    Bu fonksiyon bilgisayarında 'opencv', 'tensorflow' gibi kütüphaneler
    yüklü mü diye bakar. Yüklü değilse otomatik indirip kurar.
    Böylece 'ModuleNotFoundError' hatası almazsın.
    """
    packages = ['opencv-python', 'numpy', 'tensorflow', 'scikit-learn', 'pillow']

    print("📦 Paketler kontrol ediliyor...")
    for pkg in packages:
        try:
            # OpenCV ve Scikit-learn'ün import isimleri farklı olduğu için özel kontrol
            if pkg == 'opencv-python':
                __import__('cv2')
            elif pkg == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(pkg.replace('-', '_'))
        except ImportError:
            # Eğer kütüphane yoksa pip ile kur
            print(f"   Kuruluyor: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print("✅ Paketler hazır!\n")


# Program başladığında önce paketleri kontrol et
install_packages()

# --- GEREKLİ KÜTÜPHANELERİN İÇE AKTARILMASI ---
import cv2  # Görüntü işleme (Kamera, resim okuma)
import numpy as np  # Matematiksel işlemler ve dize (array) yönetimi
import tensorflow as tf  # Yapay zeka kütüphanesi
from tensorflow import keras  # Derin öğrenme modelleri için araçlar
from sklearn.model_selection import train_test_split  # Veriyi eğitim/test olarak bölmek için
from PIL import Image  # Resim dosyalarını açmak için (PPM formatı desteği iyi)

# --- LEVHA İSİMLERİ SÖZLÜĞÜ ---
# Model çıktı olarak bize 0, 1, 2 gibi sayılar verir.
# Bu sözlük, o sayıların hangi trafik levhası olduğunu söyler.
SIGN_NAMES = {
    0: 'Hız 20', 1: 'Hız 30', 2: 'Hız 50', 3: 'Hız 60', 4: 'Hız 70',
    5: 'Hız 80', 6: 'Hız 80 Sonu', 7: 'Hız 100', 8: 'Hız 120',
    9: 'Sollama Yasak', 10: 'Ağır Taşıt Sollama', 11: 'Kavşak',
    12: 'Ana Yol', 13: 'Yol Ver', 14: '🛑 DUR 🛑', 15: 'Araç Giremez',
    16: 'Ağır Taşıt Giremez', 17: 'Giriş Yasak', 18: 'Genel Dikkat',
    19: 'Viraj Sol', 20: 'Viraj Sağ', 21: 'Çift Viraj',
    22: 'Tümsek', 23: 'Kaygan', 24: 'Yol Daralıyor',
    25: 'Yol Çalışması', 26: 'Trafik Işığı', 27: 'Yaya',
    28: 'Okul', 29: 'Bisiklet', 30: 'Buz/Kar',
    31: 'Hayvan', 32: 'Sınır Sonu', 33: 'Sağa',
    34: 'Sola', 35: 'İleri', 36: 'İleri/Sağa',
    37: 'İleri/Sola', 38: 'Sağdan', 39: 'Soldan',
    40: 'Ada', 41: 'Sollama Sonu', 42: 'Ağır Sollama Sonu'
}


def load_gtsrb_data(img_size=48, max_per_class=1000):
    """
    --- VERİ YÜKLEME FONKSİYONU ---
    Bilgisayarındaki 'gtsrb/Training' klasörüne girer.
    0'dan 42'ye kadar olan klasörleri tek tek gezer.
    İçindeki resimleri okur, boyutlandırır ve listeye ekler.
    """
    print("\n" + "=" * 70)
    print("📂 GTSRB VERİ SETİ YÜKLENİYOR")
    print("=" * 70)
    print()

    # Klasör yolu
    base_path = 'gtsrb/Training'

    # Klasör var mı kontrolü
    if not os.path.exists(base_path):
        print(f"❌ HATA: '{base_path}' klasörü bulunamadı!")
        print()
        print("Klasör yapınız şöyle olmalı:")
        print("  gtsrb/")
        print("    └── Training/")
        print("        ├── 00000/")
        print("        ├── 00001/")
        print("        └── ...")
        print()
        raise FileNotFoundError(f"Klasör bulunamadı: {base_path}")

    print(f"✅ Veri klasörü bulundu: {base_path}")
    print()

    X_data = []  # Resimlerin kendisi buraya eklenecek
    y_data = []  # Resimlerin etiketleri (hangi levha olduğu) buraya
    total_images = 0

    # Her sınıf için (00000 - 00042 arası döngü)
    for class_id in range(43):
        class_folder = os.path.join(base_path, f'{class_id:05d}')

        if not os.path.exists(class_folder):
            print(f"⚠️  Sınıf {class_id:05d} klasörü yok, atlanıyor...")
            continue

        # Klasördeki .ppm uzantılı dosyaları bul
        ppm_files = [f for f in os.listdir(class_folder) if f.endswith('.ppm')]

        # Hepsini alırsak eğitim çok uzun sürer, 'max_per_class' kadar alıyoruz
        ppm_files = ppm_files[:max_per_class]

        class_count = 0

        for ppm_file in ppm_files:
            img_path = os.path.join(class_folder, ppm_file)

            try:
                # 1. Resmi aç
                img = Image.open(img_path)
                img = img.convert('RGB') # Renkli format

                # 2. Boyutlandır (Bütün resimler aynı boyutta olmalı, örn: 48x48)
                img = img.resize((img_size, img_size), Image.LANCZOS)

                # 3. Sayısal dizeye (array) çevir
                img_array = np.array(img)

                X_data.append(img_array)
                y_data.append(class_id)

                class_count += 1
                total_images += 1

            except Exception as e:
                # Bozuk resim varsa atla
                continue

        # Ekrana bilgi yazdır (DUR levhası önemli olduğu için onu belirgin yaz)
        sign_name = SIGN_NAMES.get(class_id, f'Sınıf {class_id}')

        if class_id == 14:  # DUR levhası
            print(f"🛑 [{class_id:05d}] {sign_name:20s} - {class_count:5d} görüntü ⭐ ÖNEMLİ!")
        else:
            print(f"   [{class_id:05d}] {sign_name:20s} - {class_count:5d} görüntü")

    print()
    print("=" * 70)
    print(f"✅ TOPLAM {total_images:,} GERÇEK LEVHA FOTOĞRAFI YÜKLENDİ!")
    print("=" * 70)
    print()

    # Listeleri numpy array formatına çevirip döndür
    return np.array(X_data, dtype='float32'), np.array(y_data, dtype='int32')


def create_cnn_model(img_size=48):
    """
    --- YAPAY ZEKA MODELİNİN TASARIMI (CNN) ---
    Bu fonksiyon, resimleri analiz edecek 'beyin' yapısını kurar.
    Katman katman (Layer) görüntü işlenir.
    """
    print("🏗️  CNN MODELİ")
    print("=" * 70)

    model = keras.Sequential([
        # 1. Normalizasyon: Pikselleri 0-255 arasından 0-1 arasına sıkıştırır. İşlemi hızlandırır.
        keras.layers.Rescaling(1. / 255, input_shape=(img_size, img_size, 3)),

        # --- BLOK 1: Kenar, köşe gibi basit özellikleri bul ---
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(), # Öğrenmeyi dengeler
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)), # Görüntü boyutunu yarıya düşürür (özetler)
        keras.layers.Dropout(0.25), # Ezberlemeyi önlemek için bazı nöronları kapatır

        # --- BLOK 2: Şekil, daire gibi orta seviye özellikleri bul ---
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # --- BLOK 3: Karmaşık desenleri bul ---
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.4),

        # --- SINIFLANDIRMA (Karar Verme) ---
        keras.layers.Flatten(), # Resmi düz bir sayı listesine çevirir
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        # Çıktı katmanı: 43 tane olasılık üretir (her levha için bir tane)
        keras.layers.Dense(43, activation='softmax')
    ])

    # Modeli derle (Hata hesaplama yöntemi ve optimizasyon algoritması seçilir)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model hazır - Parametre: {model.count_params():,}\n")
    return model


def train_model(X, y):
    """
    --- **********EĞİTİMİ YÖNETEN FONKSİYON ---
    Veriyi alır, böler, modeli oluşturur ve eğitimi başlatır.
    """
    print("=" * 70)
    print("🎓 MODEL EĞİTİMİ")
    print("=" * 70)
    print()

    # Veriyi Eğitim (%85) ve Doğrulama (%15) olarak ayır
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"📊 Eğitim: {len(X_train):,} görüntü")
    print(f"📊 Doğrulama: {len(X_val):,} görüntü\n")

    model = create_cnn_model(img_size=X.shape[1])

    # Veri Çoğaltma (Augmentation): Resimleri hafifçe döndür, kaydır, yakınlaştır.
    # Bu, modelin farklı açılardan çekilmiş levhaları da tanımasını sağlar.
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    # Callbacks: Eğitim sırasında yapılacak ekstra işlemler
    callbacks = [
        # En iyi sonucu veren modeli kaydet
        keras.callbacks.ModelCheckpoint(
            'gtsrb_real_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        # Eğer iyileşme durursa eğitimi erken bitir
        keras.callbacks.EarlyStopping(
            patience=8,
            restore_best_weights=True
        ),
        # İlerleme tıkanırsa öğrenme hızını düşür (daha hassas öğren)
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]

    print("=" * 70)
    print("⏳ EĞİTİM BAŞLADI (20-30 dakika)")
    print("=" * 70)
    print()

    # Modeli verilere oturt (Fit et)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        validation_data=(X_val, y_val),
        epochs=40,
        callbacks=callbacks,
        verbose=1
    )

    print()
    print("=" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print()

    best_acc = max(history.history['val_accuracy'])
    print(f"📊 En İyi Doğruluk: %{best_acc * 100:.2f}")
    print(f"💾 Model: gtsrb_real_model.h5\n")

    return model


class GTSRBDetector:
    """
    --- TESPİT SİSTEMİ SINIFI ---
    Bu sınıf kamerayı açar, kırmızı bölgeleri bulur ve
    eğittiğimiz modeli kullanarak levhaları tanır.
    """

    def __init__(self, model_path='gtsrb_real_model.h5'):
        print("\n🚦 GERÇEK LEVHA TESPİT SİSTEMİ")
        print("=" * 70)

        # Model dosyası var mı kontrol et
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")

        print(f"📂 Model yükleniyor...")
        self.model = keras.models.load_model(model_path)
        self.img_size = self.model.input_shape[1] # Modelin beklediği resim boyutunu al
        print(f"✅ Model hazır! (Giriş: {self.img_size}x{self.img_size})\n")

        self.threshold = 0.5 # Güven eşiği (bu değerin altındakileri görmezden gel)

    def detect_red_regions(self, frame):
        """
        --- RENK FİLTRELEME ---
        Görüntüdeki kırmızı pikselleri bulur.

        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Kırmızının iki farklı tonu vardır (Koyu ve Açık), ikisini de maskeliyoruz.
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        # İki maskeyi birleştir
        mask = mask1 | mask2

        # **************Gürültü temizleme (Küçük noktaları sil, delikleri kapat)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Maske üzerindeki şekillerin (konturların) koordinatlarını bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Çok küçük veya çok büyük alanları ele
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0

                # Kareye yakın şekilleri al (Levhalar genelde karedir/yuvarlaktır)
                if 0.7 < aspect < 1.4:
                    # Kutuyu biraz genişlet (padding) ki levha tam sığsın
                    pad = 15
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(frame.shape[1] - x, w + 2 * pad)
                    h = min(frame.shape[0] - y, h + 2 * pad)

                    regions.append((x, y, w, h))

        return regions

    def predict_region(self, region):
        """
        --- TAHMİN ETME ---
        **********Kesilen küçük resim parçasını modele gönderir ve sonucu alır.
        """
        # Resmi modelin istediği boyuta getir (örn: 48x48)
        resized = cv2.resize(region, (self.img_size, self.img_size))
        # Batch boyutunu ekle (1, 48, 48, 3)
        batch = np.expand_dims(resized, axis=0)

        # Modeli çalıştır
        predictions = self.model.predict(batch, verbose=0)[0]

        # En yüksek ihtimali bul
        class_id = np.argmax(predictions)
        confidence = predictions[class_id] # Güven oranı (0.0 - 1.0 arası)

        return class_id, confidence, predictions

    def run_camera(self):
        """
        --- KAMERA DÖNGÜSÜ ---
        Sürekli görüntü alır, işler ve ekrana basar.
        """
        print("=" * 70)
        print("🎥 KAMERA MODU")
        print("=" * 70)
        print()
        print("🎮 KONTROLLER:")
        print("   Q/ESC = Çıkış")
        print("   SPACE = Ekran görüntüsü")
        print("   -/+   = Eşik ayarı")
        print()
        print("💡levhanızı kameraya gösterin!")
        print()

        cap = cv2.VideoCapture(0) # 0, varsayılan kameradır

        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("✅ Kamera hazır!\n")

        screenshot_count = 0
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Adım: Kırmızıya benzeyen bölgeleri bul
            regions = self.detect_red_regions(frame)

            # 2. Adım: Her bölgeyi yapay zekaya sor
            for (x, y, w, h) in regions:
                roi = frame[y:y + h, x:x + w] # Region of Interest (İlgi Alanı)

                if roi.size == 0:
                    continue

                class_id, confidence, all_preds = self.predict_region(roi)

                # Eğer güven oranı eşiğin üzerindeyse işlem yap
                if confidence > self.threshold:
                    total_detections += 1

                    # Çerçeve rengini belirle
                    if class_id == 14:  # DUR
                        color = (0, 0, 255)  # Kırmızı
                        thickness = 5
                    elif confidence > 0.8:
                        color = (0, 255, 0)  # Yeşil
                        thickness = 3
                    else:
                        color = (0, 255, 255)  # Sarı
                        thickness = 2

                    # *************Kutuyu çiz
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                    # **************Metni hazırla
                    sign_name = SIGN_NAMES.get(class_id, f'ID{class_id}')
                    label = f"{sign_name} {confidence * 100:.0f}%"

                    # Yazının arkasına siyah kutu koy (okunabilirlik için)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
                    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x + 5, y - 5), font, 0.7, (255, 255, 255), 2)

                    # DUR levhası bulundu! (Opsiyonel log)
                   # if class_id == 14:
                        #print(f"🛑 DUR LEVHASI BULUNDU! Güven: %{confidence * 100:.1f}")

            # ***********Ekrana bilgi paneli (Sol üst köşe)
            info = [
                f"Esik: {self.threshold:.2f}",
                f"Aday: {len(regions)}",
                f"Toplam: {total_detections}"
            ]

            y_pos = 30
            for text in info:
                cv2.rectangle(frame, (5, y_pos - 22), (250, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30

            # Görüntüyü ekranda göster
            cv2.imshow('GTSRB Tespit - DUR levhanizi gosterin', frame)

            # Klavye kontrolü eşik değerini ayarlatyabiliyoruz
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27: # q veya ESC
                break
            elif key == ord(' '): # Boşluk tuşu (Screenshot)
                screenshot_count += 1
                fname = f"gtsrb_detection_{screenshot_count}.jpg"
                cv2.imwrite(fname, frame)
                print(f"📸 Kaydedildi: {fname}")
            elif key == ord('-'): # Hassasiyeti azalt
                self.threshold = max(0.3, self.threshold - 0.05)
                print(f"Eşik: {self.threshold:.2f}")
            elif key == ord('+'): # Hassasiyeti artır
                self.threshold = min(0.95, self.threshold + 0.05)
                print(f"Eşik: {self.threshold:.2f}")

        cap.release()
        cv2.destroyAllWindows()

        print()
        print("=" * 70)
        print("📊 ÖZET")
        print("=" * 70)
        print(f"Toplam tespit: {total_detections}")


def main():
    """
    --- ANA PROGRAM ---
    Kullanıcıya menü sunar:
    1. Kamera modunu aç
    2. Modeli eğit
    """
    print("\n" + "=" * 70)
    print("🚦 GTSRB GERÇEK LEVHA SİSTEMİ")
    print("=" * 70)
    print()

    model_file = 'gtsrb_real_model.h5'

    # Model dosyası var mı?
    if os.path.exists(model_file):
        print(f"✅ Model bulundu: {model_file}")
        print()
        print("Seçenekler:")
        print("1. Tespit sistemi başlat (Kamera)")
        print("2. Modeli yeniden eğit")
        print("3. Çıkış")
        print()

        choice = input("Seçim (1-3): ").strip()

        if choice == '1':
            detector = GTSRBDetector(model_file)
            detector.run_camera()
            return
        elif choice == '3':
            return

    # ************Eğitim Bölümü (Eğer model yoksa veya kullanıcı 2'yi seçerse)
    try:
        print("📂 Veri seti kontrol ediliyor...\n")

        # Veriyi yükle
        X, y = load_gtsrb_data(img_size=48, max_per_class=800)

        # Modeli eğit
        model = train_model(X, y)

        print("=" * 70)
        print("🎉 BAŞARILI!")
        print("=" * 70)
        print()
        print("✅ Model eğitildi ve kaydedildi!")
        print()

        # Eğitimden sonra hemen test etmek ister mi?
        if input("Şimdi test etmek ister misiniz? (E/H): ").strip().upper() == 'E':
            detector = GTSRBDetector(model_file)
            detector.run_camera()

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()


# Python dosyası direkt çalıştırılırsa main() fonksiyonunu çağır
if __name__ == "__main__":
    main()