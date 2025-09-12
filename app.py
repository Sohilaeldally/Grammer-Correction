import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    device_map=None,
    torch_dtype="auto"
)

device = "cpu"

# Function for correction
def correct_sentence(sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Custom CSS
st.markdown(
    """
    <style>
        
        .stApp {
            background-color: pink;
        }

        /* Title and headings */
        h1, h2,h3 {
            color: black !important;

        }


        /* Input label */
        label {
            color: black !important;
            font-weight: bold;
        }

        /* Input box */
        .stTextInput > div > div > input {
            background-color: white !important;
            color: black !important;
        }

        /* Button */
        div.stButton > button {
            background-color: white !important;
            color: black !important;
            font-weight: bold;
            border-radius: 6px;
            border: 1px solid black;
        }

        /* Warning / Success / Error messages */
        .stAlert {
            background-color: white !important;
            color: black !important;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.title("English Grammar Correction")
st.write("Enter a sentence and get it corrected if needed.")

# Input and correction
user_input = st.text_input("Enter sentence:")
if st.button("Correct"):
    if user_input.strip() != "":
        corrected = correct_sentence(user_input)
        st.success(f"Corrected sentence: {corrected}")
    else:
        st.warning("Please enter a sentence to correct.")
