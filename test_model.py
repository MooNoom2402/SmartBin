import tensorflow.lite as tflite
import numpy as np
import cv2

# กำหนดพาธไฟล์โมเดลและ labels
MODEL_PATH = "plastic_classifier.tflite"
LABELS_PATH = "labels.txt"
IMG_PATH = "test.jpg"  # เปลี่ยนเป็นชื่อไฟล์ภาพที่ต้องการทดสอบ

# โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# ดึงรายละเอียด input / output ของโมเดล
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1:3]  # ขนาดภาพที่ต้องใช้

# โหลด labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# โหลดและแปลงภาพให้เป็นขนาดที่โมเดลต้องการ
image = cv2.imread(IMG_PATH)
if image is None:
    print(f"❌ ไม่พบไฟล์รูป {IMG_PATH}")
    exit()

image = cv2.resize(image, tuple(IMG_SIZE))  # ปรับขนาด
image = image.astype(np.float32) / 255.0  # ทำ normalization
image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้เป็น (1, height, width, channels)

# ทำการทำนาย
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

# แสดงผลลัพธ์
predicted_index = np.argmax(predictions)  # ดึง index ของค่าที่มีค่าสูงสุด
confidence = predictions[0][predicted_index] * 100  # คำนวณเปอร์เซ็นต์ความมั่นใจ
predicted_label = labels[predicted_index]  # ดึงชื่อคลาส

print(f"🔍 ผลลัพธ์: {predicted_label} ({confidence:.2f}%)")
