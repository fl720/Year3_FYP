from flask import Flask, request, jsonify, send_from_directory, send_file , Response
from datetime import datetime
import cv2
import os
import shutil
import time
import threading
import requests
import atexit
import nn_model.comfitness as cf
import nn_model.pose_identification as pi 
from pose_model.RuleBasedFallDetector import RuleBasedFallDetector, PoseState


app = Flask(__name__)
UPLOAD_FOLDER = './cam_sc'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TEMP_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'temp.jpg')

API_KEY = '3f9f172deb25f1fcb6045cd7a82f2b1c'  
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

detector = RuleBasedFallDetector(model_path='./pose_demo/yolo11n-pose.pt')

# Create cam_sc folder if not exist
if not os.path.exists('cam_sc'):
    os.makedirs('cam_sc')

# Shared state
latest_frame = None
current_pose = "Unknown"
fall_detected = False
lock = threading.Lock()


def capture_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    TEMP_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'temp.jpg')



    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imwrite(TEMP_IMAGE_PATH, frame)

        time.sleep(0.5)  # Capture frame every 0.5 seconds

def detect_pose():
    detector = RuleBasedFallDetector(model_path='./pose_demo/yolo11n-pose.pt')

    while True:
        if not os.path.exists(TEMP_IMAGE_PATH):
            time.sleep(0.1)
            continue

        try:
            frame = cv2.imread(TEMP_IMAGE_PATH)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, pose = detector.update(frame_rgb)

            global current_pose, fall_detected
            current_pose = str(pose)
            fall_detected = (pose == PoseState.FALL)

        except Exception as e:
            print(f"[Detection Error]: {e}")

        time.sleep(0.5)  # Pose detection every 0.5 seconds

def generate_frames():
    global latest_frame

    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                   

@app.route('/video_feed')
def video_feed():
    filepath = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    else:
        # Return a blank image or 404
        return '', 404


@app.route('/pose_status')
def pose_status():
    return jsonify({
        'pose': current_pose,
        'fall': fall_detected
    })

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

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
