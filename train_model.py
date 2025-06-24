import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# 📌 กำหนดค่า Training
IMG_SIZE = (224, 224)  # ขนาดรูปภาพ
BATCH_SIZE = 32        # จำนวนรูปต่อ batch
EPOCHS = 30            # รอบการเทรน

# ✅ โหลดและเตรียม Dataset
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # แบ่ง 80% train / 20% validation
)

train_data = train_datagen.flow_from_directory(
    "dataset",  # 📂 โฟลเดอร์หลักที่มีทั้ง 12 คลาส
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ✅ โหลดโมเดล MobileNetV2 (Pretrained)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # ❌ ไม่เทรนน้ำหนักเดิม

# ✅ สร้างโมเดลใหม่
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(train_data.num_classes, activation="softmax")  # จำนวน class เท่ากับ dataset
])

# ✅ คอมไพล์โมเดล
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ ใช้ Early Stopping เพื่อหยุดเมื่อ val_loss ไม่ลดลง
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# 🚀 เทรนโมเดล
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]  # 🛑 Early Stopping
)

# ✅ เซฟโมเดล
model.save("garbage_classifier.h5")

# ✅ แปลงเป็น TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("garbage_classifier.tflite", "wb") as f:
    f.write(tflite_model)

# ✅ สร้าง labels.txt
class_indices = train_data.class_indices
labels = sorted(class_indices, key=class_indices.get)

with open("labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print("✅ โมเดลถูกเทรนเสร็จและบันทึกเรียบร้อย! 🎉")
