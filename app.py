import os
import cv2
import torch
from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Muat model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Variabel global untuk kontrol kamera (depan/belakang)
camera_index = 0  # 0 = kamera default, bisa diubah

# Fungsi untuk menangkap frame dari kamera dan mendeteksi objek
def generate_frames():
    cap = cv2.VideoCapture(camera_index)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Deteksi objek dengan YOLOv5
            results = model(frame)
            # Gambar kotak hasil deteksi pada frame
            for det in results.pred[0]:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Encode frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Rute untuk mengubah kamera
@app.route('/switch_camera')
def switch_camera():
    global camera_index
    camera_index = 1 - camera_index  # Toggle antara kamera 0 dan 1
    return redirect(url_for('index'))

# Rute untuk upload gambar dan deteksi
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join("static", filename)
        file.save(file_path)
        
        # Load gambar dan lakukan deteksi
        img = cv2.imread(file_path)
        results = model(img)
        # Gambar kotak hasil deteksi
        for det in results.pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Simpan gambar yang terdeteksi
        detected_path = os.path.join("static", "detected_" + filename)
        cv2.imwrite(detected_path, img)
        
        return redirect(url_for('static', filename="detected_" + filename))

    return 'Upload failed', 500

# Rute untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Rute untuk streaming video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
