# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

# Load pre-trained model and tokenizer for question answering
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(device)

# Context for factual Q&A (can be expanded with more information)
context = """
Pakistan is a country in South Asia. It is bordered by India, Afghanistan, Iran, and China. The country has a rich cultural heritage and diverse landscapes.
Karachi is the largest city in Pakistan and the capital of the Sindh province. It is located on the southern coast along the Arabian Sea.
Babar Azam is a Pakistani cricketer and the captain of the Pakistan national cricket team. He is widely regarded as one of the best batsmen in the world.
"""

# Function to get chatbot response based on a context and user question
def chatbot_response(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1

    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end], skip_special_tokens=True)
    return answer if answer else "Sorry, I don't know the answer to that."

# Streamlit UI
st.title("Factual Question Answering Chatbot")
st.write("Ask me a question based on the information about Pakistan, Karachi, and Babar Azam.")

# User input
user_input = st.text_input("You:")

if user_input:
    response = chatbot_response(user_input, context)
    st.write(f"Bot: {response}")




# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Function to load model and tokenizer with error handling
# @st.cache_resource
# def load_model():
#     try:
#         # Load pre-trained DialoGPT model and tokenizer
#         tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
#         model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
#         return tokenizer, model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None

# # Load model and tokenizer
# tokenizer, model = load_model()

# # Check if the model and tokenizer were successfully loaded
# if tokenizer is None or model is None:
#     st.stop()  # Stop app if model fails to load

# # Function to get chatbot response
# def chatbot_response(input_text, chat_history):
#     try:
#         # Encode the new user input, add the eos_token and return a tensor in PyTorch
#         new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

#         # Append the user input tokens to the chat history
#         chat_history_ids = chat_history + new_user_input_ids if chat_history is not None else new_user_input_ids

#         # Generate a response from the model
#         bot_output_ids = model.generate(
#             chat_history_ids, 
#             max_length=1000, 
#             pad_token_id=tokenizer.eos_token_id, 
#             do_sample=True, 
#             top_k=50, 
#             top_p=0.95, 
#             temperature=0.7
#         )

#         # Decode the bot response from the generated ids
#         bot_response = tokenizer.decode(bot_output_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
#         return bot_response, chat_history_ids
#     except Exception as e:
#         st.error(f"Error generating response: {e}")
#         return "Sorry, something went wrong. Please try again.", chat_history

# # Streamlit UI
# st.title('Hugging Face Chatbot')
# st.write("Chat with the chatbot!")

# # Initialize chat history
# chat_history = None

# # Display chat input and history
# user_input = st.text_input("You:", "")

# if user_input:
#     response, chat_history = chatbot_response(user_input, chat_history)
#     st.text_area("Bot:", value=response, height=200)

