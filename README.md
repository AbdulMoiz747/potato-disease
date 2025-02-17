Potato Disease Classification

Project Overview

This project is a machine learning-based web and mobile application that classifies potato diseases using deep learning. The system is designed to assist farmers and agricultural experts in identifying potato plant diseases through image recognition. It utilizes a Convolutional Neural Network (CNN) trained on a dataset of diseased and healthy potato leaves.

Technologies Used

Backend: FastAPI, TensorFlow, TensorFlow Serving

Frontend: React.js (Web), React Native (Mobile App)

Database: Firebase, Google Cloud Storage

Machine Learning: Python, TensorFlow, Jupyter Notebook

DevOps & Deployment: Docker, Google Cloud Platform (GCP), Google Cloud Functions

Setup Guide

1. Setting up the Environment

Backend (FastAPI & TensorFlow Serving)

Install Python and required dependencies:

pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt

Install TensorFlow Serving (Follow the Setup Instructions).

Frontend (React.js Web Application)

Install Node.js and NPM (Follow the Setup Instructions).

Install dependencies:

cd frontend
npm install --from-lock-json
npm audit fix

Copy .env.example as .env and update the API URL.

Mobile App (React Native)

Follow the React Native CLI Quickstart setup.

Install dependencies:

cd mobile-app
yarn install

For macOS users:

cd ios && pod install && cd ../

Copy .env.example as .env and update the API URL.

2. Training the Model

Download the dataset from Kaggle and keep only folders related to Potatoes.

Start Jupyter Notebook:

jupyter notebook

Open training/potato-disease-training.ipynb and update the dataset path in Cell #2.

Run all the cells sequentially to train the model.

Save the generated model with a version number in the models/ folder.

3. Running the API

Option 1: Using FastAPI

Navigate to the API folder:

cd api

Run the FastAPI server using Uvicorn:

uvicorn main:app --reload --host 0.0.0.0

The API will be available at http://0.0.0.0:8000

Option 2: Using FastAPI & TensorFlow Serving

Copy models.config.example as models.config and update the paths.

Run TensorFlow Serving:

docker run -t --rm -p 8501:8501 -v $(pwd):/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config

Run the FastAPI server:

uvicorn main-tf-serving:app --reload --host 0.0.0.0

The API will be available at http://0.0.0.0:8000

4. Running the Frontend & Mobile App

Frontend (React.js Web App)

Navigate to the frontend folder:

cd frontend

Copy .env.example to .env and update the REACT_APP_API_URL.

Start the frontend:

npm run start

Mobile App (React Native)

Navigate to the mobile app folder:

cd mobile-app

Copy .env.example to .env and update the API URL.

Run the app:

npm run android  # For Android
npm run ios      # For iOS

5. Model Deployment on Google Cloud Platform (GCP)

Deploying the TensorFlow Lite Model

Create a GCP account and set up a new project.

Create a GCP bucket and upload the potato-model.h5 file to models/potato-model.h5.

Install Google Cloud SDK and authenticate:

gcloud auth login

Deploy the model:

cd gcp
gcloud functions deploy predict_lite --runtime python38 --trigger-http --memory 512 --project project_id

The model is now deployed. Use Postman to test the GCF Trigger URL.

Deploying the TensorFlow Model (.h5) on GCP

Follow steps 1-3 above.

Upload the .h5 model to models/potato-model.h5 in your GCP bucket.

Deploy the function:

gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project project_id

The model is now deployed. Use Postman to test the GCF Trigger URL.

Inspiration & References

This project was inspired by Google Cloud’s blog on deploying deep learning models with TensorFlow 2.0. You can read more here.

For any issues or contributions, feel free to open a pull request or reach out!
