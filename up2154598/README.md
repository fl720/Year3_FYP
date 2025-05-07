# Webcam Monitoring System for High-Risk Independent Residents

This project is a real-time monitoring and alert system aimed at protecting high-risk individuals who live independently. It uses webcam input to detect falls or unusual postures, retrieves weather information to advise on health risks, and sends automated email alerts in case of emergencies.

---

## `page/` Folder Structure

The `page/` directory contains the full application  

(You can just download the `page/` folder and run the app.py with the same environment which will be listed later):

- `app.py`: Main Flask web server that serves the webpage, controls webcam input, triggers pose detection, fetches weather data, and handles email alerts.
- `static/index.html`: The frontend HTML page for the user interface.
- `static/black.jpg`: Placeholder image displayed when the webcam is turned off.
- `cam_sc/temp.jpg`: Continuously updated screenshot from the webcam.
- `pose_model.py`: Contains the logic for loading and applying the pose detection model (e.g., CNN).
- `comfitness.py`: Uses environmental data (temperature and humidity) to determine a health-related message.
- `weather_api.py`: (if applicable) Handles weather API requests and processing.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd your-repository-name/page
```


### 2. Install Dependencies
Ensure you are using Python 3.8 or later. Then install required packages:

```
pip install flask opencv-python Pillow requests python-dotenv torch numpy joblib ultralytics
```

### 3. Configure .env File
The .env file stores sensitive configuration details such as email credentials. It is required for the email alert function to work properly.

You must create a .env file in the root of the project (same level as page/) and include the following environment variables:

```
SENDER_EMAIL    = your_sender_email@example.com
SENDER_PASSWORD = your_email_password
RECEIVER_EMAIL  = recipient_email@example.com
```

⚠️ Your email credentials are never shared or uploaded to version control.

**If you are using Gmail, using an App Password instead of your real password.**

#### 🔐 Step-by-Step of how to get the App password for Gmail :
Enable 2-Step Verification (if you haven’t already)

- Go to your Google Account

- Navigate to Security > 2-Step Verification

- Enable it

Generate an App Password

- After enabling 2FA, go back to Security

- Scroll to “Signing in to Google”

- Click App passwords

- Sign in again

- Under “Select app”, choose Other (Custom) and name it like Flask Email Alert

- Click Generate

Google will give you a 16-character password

- Example: abcd efgh ijkl mnop

## Run the Application
Navigate into the page/ directory and run the app:

```
cd page
python app.py
```

Then visit the app in your browser at:

```
http://127.0.0.1:5000/
```
For personal security purpose, the web page will only be able to access locally. 

## External Tools and Dependencies
The project relies on the following external libraries and tools:

Flask – For serving the web application

OpenCV (cv2) – For capturing webcam frames and basic image processing

Requests – To fetch weather data from an external API

python-dotenv – To load sensitive environment variables from .env

HTML/CSS/JavaScript – For frontend interface and auto-refresh logic

## System Features
Live webcam feed with automatic screenshot updates every 500 ms (2FPS)

AI pose detection with real-time fall detection

Health alerts based on weather conditions (updated every 30 minutes)

"Turn Off Webcam" button to disable capture and reduce CPU usage

Automatic email alert when a fall (e.g., "Lying") is detected

---

## `source/` Folder Structure
This is the folder that contains all files that are used to develop health advisory output.

- `data/comfitness_level.py`: Using heat index function to generate correct lables for training.
- `data/comfortness_data_generate.py`: Generates correct labels from input random tempetures and random humidity levels to functions in `data/comfitness_level.py`.
- `data/comfitness_training.csv`: Labels for training
- `comfitness_level_NN.py`: Training of the feedforward nerual network.
- `test_comfitness_nn.py`: Using model to produce predict results.
- `compare_prediction.py`: Compare results from heat index function and predict results from trained nerual network. 
 

