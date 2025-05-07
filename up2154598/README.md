# Webcam Monitoring System for High-Risk Independent Residents

This project is a real-time monitoring and alert system aimed at protecting high-risk individuals who live independently. It uses webcam input to detect falls or unusual postures, retrieves weather information to advise on health risks, and sends automated email alerts in case of emergencies.

---

## `page/` Folder Structure

The `page/` directory contains the full application:

- `app.py`: Main Flask web server that serves the webpage, controls webcam input, triggers pose detection, fetches weather data, and handles email alerts.
- `templates/index.html`: The frontend HTML page for the user interface.
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


 Install Dependencies
Ensure you are using Python 3.8 or later. Then install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Alternatively, if requirements.txt is not present, install manually:

bash
Copy
Edit
pip install flask opencv-python requests python-dotenv
3. Configure .env File
Create a file named .env in the root directory (same level as page/). It is required for sending email alerts:

ini
Copy
Edit
SENDER_EMAIL=your_sender_email@example.com
SENDER_PASSWORD=your_email_password
RECEIVER_EMAIL=recipient_email@example.com
⚠️ Your email credentials are never shared or uploaded to version control.
For Gmail, consider using an App Password instead of your real password.

▶️ Run the Application
Navigate into the page/ directory and run the app:

bash
Copy
Edit
cd page
python app.py
Then visit the app in your browser at:

cpp
Copy
Edit
http://127.0.0.1:5000/
📦 External Tools and Dependencies
The project relies on the following external libraries and tools:

Flask – For serving the web application

OpenCV (cv2) – For capturing webcam frames and basic image processing

Requests – To fetch weather data from an external API

python-dotenv – To load sensitive environment variables from .env

HTML/CSS/JavaScript – For frontend interface and auto-refresh logic

🔁 System Features
Live webcam feed with automatic screenshot updates every 250 ms

AI pose detection with real-time fall detection

Health alerts based on weather conditions (updated every 30 minutes)

"Turn Off Webcam" button to disable capture and reduce CPU usage

Automatic email alert when a fall (e.g., "Lying") is detected

