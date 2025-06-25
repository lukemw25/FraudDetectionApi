Credit Card Fraud Detection API

This is a small project where I built a machine learning model to detect credit card fraud and created an API around it using FastAPI.
What's Included

    src/train_model.py – Trains a Random Forest model and saves it

    api/main.py – FastAPI app that loads the model and returns predictions

    tests/test_api.py – Basic test to check if the API works

    requirements.txt – List of required Python packages

    Dockerfile – Optional, for running the app in a container

    .gitignore – Keeps unnecessary files out of version control

    The dataset isn't uploaded because it's large. You can download it from Kaggle and place it in a data/ folder as creditcard.csv.

How to Run It
1. Set up the environment

python -m venv .venv
.venv\Scripts\activate  # Use `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt

2. Download the dataset

Find it here: 
ttps://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Get the creditcard.csv dataset from Kaggle and place it here:

data/creditcard.csv

3. Train the model

python src/train_model.py

4. Start the API

uvicorn api.main:app --reload

Then open your browser and go to http://localhost:8000/docs to test it.
Why I Made This

I wanted to practice building and testing a basic machine learning pipeline, and also learn how to serve a model using FastAPI.