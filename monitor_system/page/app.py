from flask import Flask, request, jsonify, send_from_directory, send_file , Response
from datetime import datetime
import cv2
import os
import time
import threading
import requests
import nn_model.comfitness as cf
from pose_model.RuleBasedFallDetector import RuleBasedFallDetector, PoseState


app = Flask(__name__)

UPLOAD_FOLDER = './cam_sc'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TEMP_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
# Shared state
latest_frame = None
current_pose = "Unknown"
fall_detected = False
lock = threading.Lock()
webcam_on = True
last_saved = 0

# detector = RuleBasedFallDetector(model_path='./pose_demo/yolo11n-pose.pt')

# Create cam_sc folder if not exist
if not os.path.exists('./page/cam_sc'):
    os.makedirs('./page/cam_sc')

API_KEY = '3f9f172deb25f1fcb6045cd7a82f2b1c'  
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

def capture_camera():
    cap = cv2.VideoCapture(0)  # Webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    print("[INFO] Webcam stream started")

    last_saved_time = time.time()  # Track time for 2 FPS

    while webcam_on:
        ret, frame = cap.read()
        if not ret:
            break

        # Save image every 0.5 seconds
        current_time = time.time()
        if current_time - last_saved_time >= 0.5:
            cv2.imwrite(TEMP_IMAGE_PATH, frame)  # Save image
            print("[INFO] temp.jpg saved.")

            last_saved_time = current_time

        time.sleep(0.01)  # Prevent tight loop

    cap.release()
    print("[INFO] Webcam stream stopped")

    # Delete image if it still exists
    if os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)
        print("[INFO] temp.jpg deleted.")

def detect_pose():
    global latest_frame, current_pose, fall_detected
    detector = RuleBasedFallDetector(model_path='./pose_demo/yolo11n-pose.pt')

    while True:
        if not webcam_on:
            time.sleep(0.5)
            continue

        try:
            if os.path.exists(TEMP_IMAGE_PATH):
                frame = cv2.imread(TEMP_IMAGE_PATH)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, pose = detector.update(frame_rgb)

                with lock:
                    current_pose = str(pose)
                    fall_detected = (pose == PoseState.FALL)
            else:
                with lock:
                    current_pose = "Waiting for frame..."

        except Exception as e:
            print(f"[Detection Error]: {e}")
            with lock:
                current_pose = "Error"
        
        time.sleep(0.5)


def generate_frames():
    global latest_frame

    # while True:
    #     if not webcam_on:
    #         time.sleep(0.1)
    #         continue

    #     with lock:
    #         frame = latest_frame.copy() if latest_frame is not None else None

    #     if frame is None:
    #         time.sleep(0.1)
    #         continue

    #     ret, buffer = cv2.imencode('.jpg', frame)
    #     if not ret:
    #         continue

    #     frame_bytes = buffer.tobytes()

    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # ---- NEW TYPE ----
    while True:
        if os.path.exists(TEMP_IMAGE_PATH):
            with open(TEMP_IMAGE_PATH, 'rb') as f:
                frame_bytes = f.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.5)

                   

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_pose')
def get_pose():
    global current_pose, fall_detected
    return jsonify({
        'pose': current_pose,
        'fall': fall_detected
    })

@app.route('/toggle_webcam', methods=['POST'])
def toggle_webcam():
    global webcam_on, current_pose

    webcam_on = not webcam_on

    if not webcam_on:
        current_pose = "Webcam Off"  # Show this when webcam is turned off

    return jsonify({'webcam_on': webcam_on})

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return '', 204

@app.route('/weather')
def weather():
    city = request.args.get('city')
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if city:
        params = {
            'q': city,
            'appid': API_KEY,
            'units': 'metric'
        }
    elif lat and lon:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': 'metric'
        }
    else:
        return jsonify({'error': 'City or coordinates required'}), 400

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if response.status_code != 200:
        return jsonify({'error': data.get('message', 'Error fetching weather')}), response.status_code

    temp = data['main']['temp']
    humidity = data['main']['humidity']
    comfitness_instance = cf.Comfitness()
    message = comfitness_instance.getComfitness(temp, humidity)
    
    return jsonify({
        'temperature': temp,
        'humidity': humidity,
        'message': message
    })

@app.route('/cam_sc/<path:filename>')
def serve_cam_sc(filename):
    return send_from_directory('page/cam_sc', filename)




if __name__ == '__main__':
    threading.Thread(target=capture_camera, daemon=True).start()
    threading.Thread(target=detect_pose, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
