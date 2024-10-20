import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to get chatbot response
def chatbot_response(input_text, chat_history):
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = chat_history + new_user_input_ids if chat_history is not None else new_user_input_ids
    bot_output_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    bot_response = tokenizer.decode(bot_output_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response, chat_history_ids

# Streamlit UI
st.title('Hugging Face Chatbot')
st.write("Chat with the chatbot!")

# Initialize chat history
chat_history = None

# Display chat input and history
user_input = st.text_input("You:", "")

if user_input:
    response, chat_history = chatbot_response(user_input, chat_history)
    st.text_area("Bot:", value=response, height=200)

