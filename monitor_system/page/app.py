from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os
import requests
import nn_model.comfitness as cf
import nn_model.pose_identification as pi 

app = Flask(__name__)
UPLOAD_FOLDER = './page'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_KEY = '3f9f172deb25f1fcb6045cd7a82f2b1c'  

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
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}'
    elif lat and lon:
        url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}'
    else:
        return jsonify({'error': 'Location required'}), 400

    try:
        res = requests.get(url).json()
        temp = res['main']['temp']
        humidity = res['main']['humidity']
        message = cf.Comfitness.getComfitness(temp, humidity)
        return jsonify({
            'temperature': temp,
            'humidity': humidity,
            'message': message
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
