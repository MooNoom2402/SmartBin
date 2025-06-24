from flask import Flask, request
import cv2
import numpy as np
import tensorflow.lite as tflite

app = Flask(__name__)

# ✅ โหลดโมเดล AI
interpreter = tflite.Interpreter(model_path="garbage_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)

@app.route('/predict', methods=['GET'])
def predict():
    # อ่านภาพจากกล้อง DroidCam หรือ ESP32-CAM
    url = "http://192.168.143.43:4747/video"  # เปลี่ยนเป็น IP ของ DroidCam
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "error"

    # แปลงภาพให้ตรงกับโมเดล
    image = cv2.resize(frame, IMG_SIZE)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0

    # รันโมเดล
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # คืนค่าผลลัพธ์
    predicted_index = np.argmax(output_data)
    labels = ["battery", "biological", "brown-glass", "cardboard", "clothes", 
              "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]

    return labels[predicted_index]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
