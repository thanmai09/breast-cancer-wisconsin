# app.py (snippet)
import os
import urllib.request
import streamlit as st
import tensorflow as tf
import pickle

MODEL_LOCAL = "BC_model.h5"
SCALER_LOCAL = "scaler.pkl"

def download_file(url, dest):
    urllib.request.urlretrieve(url, dest)

@st.cache_data
def load_artifacts():
    # 1) model
    if not os.path.exists(MODEL_LOCAL):
        model_url = st.secrets.get("MODEL_URL")  # set this in Streamlit secrets
        if model_url:
            st.info("Downloading model...")
            download_file(model_url, MODEL_LOCAL)
        else:
            raise FileNotFoundError(
                f"{MODEL_LOCAL} not found. Add the file to the repo or set MODEL_URL in secrets."
            )

    # 2) scaler
    if not os.path.exists(SCALER_LOCAL):
        scaler_url = st.secrets.get("SCALER_URL")
        if scaler_url:
            st.info("Downloading scaler...")
            download_file(scaler_url, SCALER_LOCAL)
        else:
            raise FileNotFoundError(
                f"{SCALER_LOCAL} not found. Add the file to the repo or set SCALER_URL in secrets."
            )

    model = tf.keras.models.load_model(MODEL_LOCAL)
    with open(SCALER_LOCAL, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler
