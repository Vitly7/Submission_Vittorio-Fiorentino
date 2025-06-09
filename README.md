
# Laporan Klasifikasi Gambar Hewan - Vittorio Fiorentino

---

## Domain Proyek

Klasifikasi gambar merupakan salah satu aplikasi penting dalam bidang visi komputer, di mana sistem mampu mengenali dan mengelompokkan objek dari gambar secara otomatis. Dalam dunia nyata, kemampuan untuk membedakan jenis hewan seperti anjing (dog), kucing (cat), dan ular (snake) bisa sangat berguna, misalnya untuk aplikasi keamanan, pemantauan hewan liar, atau sistem pencarian otomatis berbasis gambar.

Dengan kemajuan teknologi deep learning, terutama penggunaan Convolutional Neural Networks (CNN), performa klasifikasi gambar telah meningkat secara signifikan. CNN mampu mengekstraksi fitur visual kompleks secara otomatis tanpa perlu ekstraksi fitur manual seperti pada metode tradisional. Model-model ini telah terbukti efektif dalam tugas-tugas klasifikasi gambar, termasuk pengenalan hewan.

---

## Business Understanding

### Problem Statements
Bagaimana membangun model klasifikasi gambar untuk membedakan antara anjing, kucing, dan ular?

Algoritma deep learning mana yang memberikan performa terbaik dalam klasifikasi gambar hewan ini?

### Goals
Mengembangkan model klasifikasi gambar yang mampu mengenali jenis hewan dari gambar input (dog, cat, snake).

Membandingkan performa beberapa arsitektur deep learning seperti Conv2D dan transfer learning.

### Solution Statement

Untuk menyelesaikan permasalahan klasifikasi gambar hewan ini, digunakan dua pendekatan berbasis deep learning, yaitu:

Model CNN dengan Conv2D:
Model dibuat dari awal menggunakan layer Conv2D, MaxPooling, Flatten, dan Dense. Pendekatan ini memungkinkan pemahaman mendalam terhadap proses training dan ekstraksi fitur dari gambar. Model ini akan dilatih dengan data gambar anjing, kucing, dan ular untuk mengenali pola visual khas dari masing-masing kelas.

Transfer Learning:
Menggunakan arsitektur pretrained seperti MobileNetV2 atau ResNet50, model memanfaatkan bobot hasil pelatihan pada dataset besar (misalnya ImageNet) untuk mengekstraksi fitur. Layer atas (fully connected) disesuaikan dengan klasifikasi 3 kelas (dog, cat, snake). Transfer learning mempercepat pelatihan dan meningkatkan akurasi, terutama saat data training terbatas.

Evaluasi kedua model dilakukan menggunakan akurasi, precision, recall, dan F1-score pada data pengujian, untuk mengetahui pendekatan mana yang paling efektif dalam mengklasifikasikan gambar hewan.


---

## Data Understanding

- Sumber Data: https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset

Dataset diperoleh dari kaggle dan langsung diimport: animal-image-classification-dataset.zip

Dataset yang digunakan merupakan data gambar hewan: dogs, cats, snakes dengan atribut:
- Jumlah gambar dogs = 1000 gambar
- Jumlah gambar cats = 1000 gambar
- Jumlah gambar snakes = 1000 gambar

### Data Preprocessing

**Split Dataset**
Dataset dipisah ke folder train dan test lalu gabung ke dalam folder dataset yang baru untuk memudahkan dalam klasifikasi tiap hewan.

**Plot Gambar Hewan untuk Semua Kelas**

![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/plothewan.png)

**Plot Distribusi Untuk Semua Kelas**
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/plotkelas.png)
---

### Exploratory Data Analysis (EDA)

#### Data Augmentasi

transformations = {
    'rotate anticlockwise': anticlockwise_rotation, Berfungsi: (Memutar gambar berlawanan arah jarum jam)
    'rotate clockwise': clockwise_rotation, Berfungsi: (Memutar gambar searah jarum jam)
    'warp shift': warp_shift, Berfungsi: (Melakukan pergeseran bentuk secara tidak merata (distorsi))
    'blurring image': blur_image, Berfungsi: (Mengaburkan gambar untuk mensimulasikan noise kamera atau gerakan (motion blur))
    'add brightness': add_brightness, Berfungsi: (Meningkatkan pencahayaan gambar)
    'flip up down': flip_up_down, Berfungsi: (Membalik gambar secara vertikal)
    'shear image': sheared Berfungsi: (Menggeser bagian atas/bawah gambar ke samping secara diagonal (shear transform))

- Proses augmentasi berguna untuk memproses gambar agar memudahkan dalam proses analisa

Lalu kita akan memisahkan data asli dengan data yang telah di augmentasi

![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/augmentasi.png)


---

## Data Preparation

**Splitting Data: Train dan Test**

**Pembagian data training dan test adalah 80:20**

Distribusi data:
labels
cats                1000
cats_augmented       200
dogs                1000
dogs_augmented       200
snakes              1000
snakes_augmented     200
dtype: int64
Train size: 2880
Test size: 720

**Setelah itu kita akan menggabungkan dataframe sehingga menjadi seperti ini**

set    labels          
test   cats                188
       cats_augmented       48
       dogs                212
       dogs_augmented       37
       snakes              189
       snakes_augmented     46
train  cats                812
       cats_augmented      152
       dogs                788
       dogs_augmented      163
       snakes              811
       snakes_augmented    154

---

### Image Data Generator

**Membuat path untuk masing-masing kelas:**

train_dogs = os.path.join(TRAIN_DIR, 'dogs')
train_cats = os.path.join(TRAIN_DIR, 'cats')
train_snakes = os.path.join(TRAIN_DIR, 'snakes')

test_dogs = os.path.join(TEST_DIR, 'dogs')
test_cats = os.path.join(TEST_DIR, 'cats')
test_snakes = os.path.join(TEST_DIR, 'snakes')

Output:

Total number of dog images in training set:  951
Total number of cat images in training set:  964
Total number of snake images in training set:  965
Total number of dog images in test set:  249
Total number of cat images in test set:  236
Total number of snake images in test set:  235

**Setelah itu kita akan membuat ImageDataGenerator untuk normalisasi dan split validasi dari training set:**

Found 2305 images belonging to 3 classes.
Found 575 images belonging to 3 classes.
Found 720 images belonging to 3 classes.

---

## Modeling

Dua model machine learning yang digunakan:
- **Conv2D**

**CNN Architecture Using 128 Neurons in Conv Layer**
Penggunaan model Conv2D menggunakan Convolutional layer, Batch Normalization layer, and Pooling layer
 - Total params: 6,387,331 (24.37 MB)
 - Trainable params: 6,386,563 (24.36 MB)
 - Non-trainable params: 768 (3.00 KB)

 history_3 = model_3.fit(
    train_generator,
    epochs=30,
    batch_size=32,
    validation_data=validation_generator,
    class_weight=class_weights
)

Setelah mencoba menguji modelling Conv2D, dengan epoch=30, didapatkan hasil yang kurang memuaskan.

Dikarenakan uji coba modelling Conv2D, cukup rendah dengan nilai accuracy sekitar 0.5301, maka kita akan menggunakan transfer learning untuk meningkatkan nilai accuracy.
accuracy: 0.5301 - loss: 0.9218 - val_accuracy: 0.4974 - val_loss: 0.9357


- **Transfer Learning**

Oleh sebab itu, kita akan menggunakan model Transfer Learning.

Transfer Learning merupakan teknik menggunakan model deep learning yang sudah dilatih sebelumnya (seperti MobileNetV2) pada dataset besar seperti ImageNet, lalu menyesuaikannya (fine-tuning) untuk klasifikasi seperti hewan yang spesifik.

Pada uji coba model ini akan menggunakan Convolutional layer, Batch Normalization layer, and Pooling layer yang dibungkus oleh model transfer learning.

model = Sequential([
    Input(shape=(150, 150, 3)),

    # Pre-trained MobileNetV2 sebagai feature extractor
    MobileNetV2(include_top=False, weights='imagenet', input_shape=(150, 150, 3)),

    # Tambahan Conv2D dan MaxPooling setelah MobileNetV2
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

 Total params: 2,643,555 (10.08 MB)
 Trainable params: 2,609,443 (9.95 MB)
 Non-trainable params: 34,112 (133.25 KB)

Dari uji coba modelling Transfer Learning, terlihat nilai accuracy meningkat sampai 0,9270.

accuracy: 0.9270 - loss: 0.1389 - val_accuracy: 0.9513 - val_loss: 0.2487

---

## Evaluation

### Plot Accuracy dan Loss pada Conv2D

**Conv2D**

Visual Accuracy
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/acc2d.png)

Visual Loss
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/loss2d.png)

Visual Correlation Matrix
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/matrix2d.png)

Classification Report:
              precision    recall  f1-score   support

        cats     1.0000    0.0077    0.0153       260
        dogs     0.3986    0.9569    0.5627       232
      snakes     0.9068    0.6404    0.7506       228

    accuracy                         0.5139       720
   macro avg     0.7685    0.5350    0.4429       720
weighted avg     0.7767    0.5139    0.4245       720

---
### Plot Accuracy dan Loss pada Transfer Learning

**Transfer Learning**

Visual Accuracy
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/acctf.png)

Visual Loss
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/losstf.png)

Visual Correlation Matrix
![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/matrixtf.png)

Classification Report:

              precision    recall  f1-score   support

        cats     0.9664    0.8846    0.9237       260
        dogs     0.9524    0.8621    0.9050       232
      snakes     0.8382    1.0000    0.9120       228

    accuracy                         0.9139       720
   macro avg     0.9190    0.9156    0.9136       720
weighted avg     0.9213    0.9139    0.9140       720


---

### Hasil Pengujian Data


Evaluasi Akurasi Prediksi:

Dengan modelling Conv2D awalnya mendapat accuracy yang cukup rendah, tetapi saat menggunakan pendekatan transfer learning, nilai rata-rata accuracy meningkat menjadi 0.9139 (Hasil epoch terakhir bernilai 0.9270)

- Conv2D

accuracy: 0.5301 - loss: 0.9218 - val_accuracy: 0.4974 - val_loss: 0.9357

- Transfer Learning

accuracy: 0.9270 - loss: 0.1389 - val_accuracy: 0.9513 - val_loss: 0.2487


---

### Konversi Model

Model akan di konversi ke dalam tiga jenis format. 

- Format SavedModel (Format default TensorFlow untuk menyimpan model lengkap dan deployment di server/Cloud )
- Format TFJS (Diperlukan untuk deployment ke web apps (running di browser tanpa server))
- Format TF-Lite (Format model TensorFlow yang dikompresi dan dioptimalkan untuk perangkat mobile atau embedded)

Opsional: Kemudian hasil konversi akan disimpan ke dalam folder files.download('koko_model_all.zip')
Berfungsi untuk mendownload model

---

### Inference

Hasil inference bertujuan untuk  uji coba prediksi model menggunakan website.

![ss4](https://github.com/Vitly7/Submission_Vittorio-Fiorentino/blob/382a5a331f67b927565902ee759e16d131234da0/Gambar/inference.png)

