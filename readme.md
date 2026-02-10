# Cervical Spine AI Analyzer

The Cervical Spine AI Analyzer is a web based application that performs automated bone segmentation and analysis on cervical spine X ray images. The system allows users to upload X ray images, run AI based segmentation, and view quantitative analysis results through a simple and guided interface.

This application is designed as a clinical decision support prototype and demonstrates the integration of computer vision, machine learning, and web technologies.

---

## Features

- Secure login system
- Upload and analyze cervical spine X ray images
- AI based bone segmentation using a pretrained deep learning model
- Visual output including segmentation mask and overlay image
- Quantitative analysis such as pixel count and percentage of bone regions
- Downloadable results including images and statistics
- Deployed as a cloud hosted web application

---

## Technology Stack

- Python 3
- Flask web framework
- PyTorch for deep learning inference
- NumPy for numerical processing
- Pillow for image handling
- Gunicorn as the production WSGI server
- Render for cloud deployment

---

## Project Structure

.
├── app.py # Flask application entry point
├── routes.py # Application routes and logic
├── auth.py # Authentication logic
├── model_handler.py # Model loading and inference logic
├── model/ # Pretrained model files
├── templates/ # HTML templates
├── static/ # CSS and JavaScript assets
├── requirements.txt # Python dependencies
└── README.md


---

## How the System Works

1. User logs into the system
2. User uploads a cervical spine X ray image
3. The image is temporarily stored in a runtime directory
4. The AI model performs bone segmentation
5. Segmentation results and statistics are generated
6. Results are displayed on the results page and can be downloaded

---

## Deployment Notes

This application is deployed on Render as a Python web service.

Important deployment considerations:

- Uploaded images and generated results are stored in a temporary runtime directory
- The application automatically detects the cloud environment and adapts file handling accordingly
- Static directories are treated as read only in production
- The AI model is loaded once at application startup for efficiency

The production server is started using:

gunicorn app:app --timeout 120


---

## Local Development Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

pip install -r requirements.txt


4. Run the application

python app.py


5. Open the browser at

http://127.0.0.1:5000


---

## Model Information

The system uses a pretrained deep learning model for cervical spine bone segmentation. The model performs pixel level classification to identify anatomical bone regions and generates both visual and numerical outputs.

---

## Disclaimer

This application is intended for educational and research purposes only. It is not a certified medical device and should not be used for clinical diagnosis or treatment decisions.

---
Version's Date: 10/02/2026

