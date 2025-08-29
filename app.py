import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load model
@st.cache_resource
def get_model():
    model = load_model("best_model.keras")
    return model

model = get_model()

# Load idx2char mapping
with open("idx2char.pkl", "rb") as f:
    idx2char = pickle.load(f)

with open("char2idx.pkl", "rb") as f:
    char2idx = pickle.load(f)

model = tf.keras.models.load_model("best_model.keras", compile=False)
def generate_text(model, start_string, num_generate=2, temperature=1.0):
    input_eval= [char2idx.get(s, 0) for s in start_string.lower()]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.layers[1].reset_states()  # Reset LSTM layer states

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


# Title
st.title("🔮 My Text Generation App")

# User input
user_input = st.text_input("Enter some text:")

if st.button("Predict"):
        predicted_text=generate_text(model, start_string=user_input, num_generate=200, temperature=0.8)
        
        st.write("### Prediction Output:")
        st.text(predicted_text)

