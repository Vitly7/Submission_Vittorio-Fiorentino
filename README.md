
# Laporan Klasifikasi Gambar Hewan - Vittorio Fiorentino

---

## Domain Proyek

Pergerakan harga saham menjadi aspek penting dalam dunia keuangan, baik bagi investor individu maupun institusional. Dalam era digital saat ini, prediksi harga saham berbasis data menjadi semakin diminati karena mampu memberikan insight untuk pengambilan keputusan yang lebih akurat. Dengan meningkatnya volume data keuangan dan kemampuan komputasi, pemanfaatan machine learning untuk menganalisis pola harga saham menjadi solusi potensial yang efektif dan efisien.

Menurut [Fischer & Krauss, 2018], model deep learning seperti LSTM dapat mengalahkan model tradisional dalam prediksi harga saham harian. Hal ini memperkuat urgensi dan relevansi topik ini dalam konteks bisnis dan teknologi modern.


---

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi harga penutupan saham berdasarkan data historis harga saham?
2. Model machine learning mana yang memberikan performa terbaik dalam memprediksi harga saham?

### Goals

1. Menghasilkan model yang mampu memprediksi harga saham berdasarkan fitur-fitur seperti Open, High, Low, dan Volume.
2. Membandingkan beberapa algoritma machine learning untuk mengetahui performa terbaik.

### Solution Statement

Untuk mencapai tujuan di atas, digunakan tiga algoritma machine learning, yaitu:
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Gradient Boosting

Evaluasi model dilakukan menggunakan Root Mean Squared Error (RMSE) pada data pelatihan dan pengujian.


---

## Data Understanding
Dataset diperoleh dari kaggle dan langsung diimport: animal-image-classification-dataset.zip
Dataset yang digunakan merupakan data gambar hewan: dogs, cats, snakes dengan atribut:
- Jumlah gambar dogs = 1000 gambar
- Jumlah gambar cats = 1000 gambar
- Jumlah gambar snakes = 1000 gambar

---

### Exploratory Data Analysis (EDA)

#### Data Augmentasi

transformations = {
    'rotate anticlockwise': anticlockwise_rotation,
    'rotate clockwise': clockwise_rotation,
    'warp shift': warp_shift,
    'blurring image': blur_image,
    'add brightness': add_brightness,
    'flip up down': flip_up_down,
    'shear image': sheared

- Proses augmentasi berguna untuk memproses gambar agar memudahkan dalam proses analisa

Lalu kita akan memisahkan data asli dengan data yang telah di augmentasi


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

Dikarenakan uji coba modelling Conv2D, cukup rendah dengan nilai accuracy sekitar 0.6842, maka kita akan menggunakan transfer learning untuk meningkatkan nilai accuracy.
accuracy: 0.6574 - loss: 0.6931 - val_accuracy: 0.6957 - val_loss: 0.6488

- **Transfer Learning**

Dari uji coba modelling Transfer Learning, terlihat nilai accuracy meningkat sampai 0,9121.
accuracy: 0.9121 - loss: 0.1624 - val_accuracy: 0.9583 - val_loss: 0.2578

---

## Evaluation

### Metrik Evaluasi

**Conv2D**



**Transfer Learning**



### Hasil Pengujian Data

|    y_true     | prediksi_KNN   | prediksi_RF  | prediksi_Boosting |
| 	            |                |              |                   |   
|   270000.0    |   318473.2     |   320560.6   |    382082.5       |

Evaluasi Akurasi Prediksi:

Hasil Asli = 270000

- KNN 334973 -> Cukup dekat
- RandomForest 324502.9 -> Paling dekat
- Boosting 383839.0 -> Cukup jauh dari aslinya

---

### Kesimpulan:
- Semua model cukup baik kali ini, terutama Random Forest yang prediksinya hampir identik dengan nilai sebenarnya.

- Model Boosting cenderung underestimate (meremehkan nilai).

- KNN juga cukup akurat, hanya sedikit lebih tinggi.
