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
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

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

if __name__ == '__main__':
    app.run(debug=True)
