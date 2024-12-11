import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import random

# Load models and tokenizers with caching
@st.cache_resource
def load_dialogpt():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

@st.cache_resource
def load_distilbert():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

dialogpt_tokenizer, dialogpt_model = load_dialogpt()
distilbert_tokenizer, distilbert_model = load_distilbert()

# Function to generate a response using DialoGPT
def generate_response(prompt, chat_history_ids=None, max_length=100):
    inputs = dialogpt_tokenizer.encode(prompt + dialogpt_tokenizer.eos_token, return_tensors="pt")
    inputs = torch.cat([chat_history_ids, inputs], dim=-1) if chat_history_ids is not None else inputs

    outputs = dialogpt_model.generate(
        inputs,
        max_length=max_length,
        pad_token_id=dialogpt_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9
    )
    response = dialogpt_tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response, outputs

# Function to analyze emotion using DistilBERT
def analyze_emotion(text):
    inputs = distilbert_tokenizer(text, return_tensors="pt")
    outputs = distilbert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    return sentiment, confidence

# App Layout
st.title("AI Companion Agent")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a feature", ["Chat", "Storytelling", "Emotion Support"])

# Chat feature
if option == "Chat":
    st.header("Chat with your Companion")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_input = st.text_input("Say something:")
    if user_input:
        response, st.session_state.chat_history = generate_response(user_input, st.session_state.chat_history)
        emotion, confidence = analyze_emotion(user_input)
        st.write(f"**AI Companion:** {response}")
        st.write(f"*Detected Emotion: {emotion} (Confidence: {confidence:.2f})*")

# Storytelling feature
elif option == "Storytelling":
    st.header("Let me tell you a story!")
    stories = [
        "Once upon a time, in a forest filled with magical creatures, there lived a brave squirrel named Max...",
        "In a small village by the sea, a young girl named Ella discovered a mysterious seashell...",
        "Long ago, in a kingdom far away, a kind king made a wish that changed everything..."
    ]
    if st.button("Tell me a story"):
        st.write(random.choice(stories))

# Emotion Support feature
elif option == "Emotion Support":
    st.header("Emotional Support")
    mood = st.selectbox("How are you feeling?", ["Happy", "Sad", "Anxious", "Angry", "Neutral"])
    responses = {
        "Happy": "That's great! Keep spreading the positivity!",
        "Sad": "I'm here for you. Remember, tough times don't last.",
        "Anxious": "Take a deep breath. Try this: inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds.",
        "Angry": "It's okay to feel angry. Let's try to calm down together.",
        "Neutral": "I'm here to chat or help however you need!"
    }
    if mood:
        st.write(responses[mood])

# Footer
st.write("---")

