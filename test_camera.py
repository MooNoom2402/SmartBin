import cv2
import numpy as np
import tensorflow.lite as tflite

# ✅ โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path="garbage_classifier.tflite")
interpreter.allocate_tensors()

# ✅ โหลด labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ✅ ดึงข้อมูลของ tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)  # ขนาดภาพที่ใช้ในโมเดล

# 📷 **ตั้งค่า URL ของ DroidCam**
url = "http://192.168.107.43:4747/video"  # 🔄 เปลี่ยนเป็น IP ของ DroidCam ของคุณ
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ ไม่สามารถเชื่อมต่อกับกล้อง DroidCam ได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # 🔄 แปลงขนาดภาพให้ตรงกับโมเดล
    image = cv2.resize(frame, IMG_SIZE)
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้เป็น (1, 224, 224, 3)
    image = image.astype(np.float32) / 255.0  # Normalize 0-1

    # 🎯 ใส่ภาพเข้าไปในโมเดล
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # 📊 ดึงผลลัพธ์
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)  # หาหมวดหมู่ที่ค่ามากที่สุด
    confidence = np.max(output_data) * 100  # คำนวณความมั่นใจ (%)

    label = labels[predicted_index]
    text = f"{label} ({confidence:.2f}%)"

    # 🖍️ แสดงผลลัพธ์บนวิดีโอ
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Plastic Detection (DroidCam)", frame)

    # กด 'q' เพื่อปิดกล้อง
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
