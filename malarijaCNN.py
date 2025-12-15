
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef
)


SOURCE_DIR = "/kaggle/input/cell-images-for-detecting-malaria/cell_images"
TARGET_DIR = "/kaggle/working/MalariaSplit"
classes = ["Parasitized", "Uninfected"]
splits = {"train": 0.75, "val": 0.15, "test": 0.1}

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

for cls in classes:
    files = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * splits["train"])
    n_val = int(n_total * splits["val"])

    for i, f in enumerate(tqdm(files, desc=f"Copying {cls}")):
        src = os.path.join(SOURCE_DIR, cls, f)
        if i < n_train:
            dst = os.path.join(TARGET_DIR, "train", cls, f)
        elif i < n_train + n_val:
            dst = os.path.join(TARGET_DIR, "val", cls, f)
        else:
            dst = os.path.join(TARGET_DIR, "test", cls, f)
        shutil.copy(src, dst)


for split in ["train", "val", "test"]:
    for cls in classes:
        path = os.path.join(TARGET_DIR, split, cls)
        print(f"{split}/{cls}: {len(os.listdir(path))} slika")


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
SEED = 42

train_dir = os.path.join(TARGET_DIR, "train")
val_dir = os.path.join(TARGET_DIR, "val")
test_dir = os.path.join(TARGET_DIR, "test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=True, seed=SEED
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=VAL_BATCH_SIZE,
    class_mode='binary', shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)


base_model = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.load_weights("/kaggle/input/vgg19-weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
for layer in base_model.layers:
    layer.trainable = False

inputs = Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization(momentum=0.9)(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization(momentum=0.9)(x)
x = Dropout(0.4)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs, name="VGG19_Malaria_MCNN")

model.compile(
    optimizer=Adamax(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Treniranje sa EarlyStopping i ReduceLROnPlateau
SAVE_DIR = "/kaggle/working"
os.makedirs(SAVE_DIR, exist_ok=True)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
checkpoint_path = os.path.join(SAVE_DIR, "best_vgg19_malaria_model.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop, lr_reduce, checkpoint],
    verbose=1
)

#  Grafici za osnovno treniranje
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy tokom treniranja')
plt.xlabel('Epoh'); plt.ylabel('Tačnost'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss tokom treniranja')
plt.xlabel('Epoh'); plt.ylabel('Gubitak'); plt.legend()
plt.tight_layout(); plt.show()

# Fine-tuning
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print(f"Ucitane tezine iz {checkpoint_path}")
for layer in base_model.layers:
    if layer.name.startswith("block5"):
        layer.trainable = True
model.compile(optimizer=Adamax(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

ft_ckpt = os.path.join(SAVE_DIR, "best_vgg19_malaria_finetuned.h5")
ft_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ft_ckpt_cb = ModelCheckpoint(ft_ckpt, monitor='val_accuracy', save_best_only=True, verbose=1)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[ft_stop, lr_reduce, ft_ckpt_cb],
    verbose=1
)

# Grafici za fine-tuning
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history_ft.history['accuracy'], label='Train accuracy (FT)')
plt.plot(history_ft.history['val_accuracy'], label='Validation accuracy (FT)')
plt.title('Accuracy tokom fine-tuninga')
plt.xlabel('Epoh'); plt.ylabel('Tačnost'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history_ft.history['loss'], label='Train loss (FT)')
plt.plot(history_ft.history['val_loss'], label='Validation loss (FT)')
plt.title('Loss tokom fine-tuninga')
plt.xlabel('Epoh'); plt.ylabel('Gubitak'); plt.legend()
plt.tight_layout(); plt.show()

# Evaluacija
if os.path.exists(ft_ckpt):
    model.load_weights(ft_ckpt)
    print(f"Evaluacija sa fine-tuned modelom: {ft_ckpt}")
elif os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print(f"Evaluacija sa baznim modelom: {checkpoint_path}")

test_gen.reset()
probs = model.predict(test_gen, verbose=0).ravel()
y_pred = (probs >= 0.5).astype(int)
y_true = test_gen.classes

class_indices = test_gen.class_indices
class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, probs)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp)
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Specificity: {spec:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"ROC-AUC:     {auc:.4f}")
print(f"MCC:         {mcc:.4f}")

# Konfuziona matrica
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.show()

#  ROC kriva
fpr, tpr, _ = roc_curve(y_true, probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(); plt.tight_layout(); plt.show()
