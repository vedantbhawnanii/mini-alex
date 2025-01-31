import streamlit as st
import os
from bot import main  # Assuming 'bot.py' contains your chat logic

st.title("Simple Chat with Gemini API")

# Initialize session state for API key if not already set
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None

# Function to handle the API key input and save to session state
def handle_api_key():
    api_key = st.text_input("Enter your Gemini API Key:", type="password")
    if api_key:
        st.session_state.gemini_api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key #set environment variable
        st.success("API Key saved!")
    else:
        st.warning("Please enter your API Key.")

# Check if API key is available, otherwise ask for it
if not st.session_state.gemini_api_key:
   handle_api_key()
else:
    # Initialize messages only if API key is provided
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and response
    if prompt := st.chat_input("What are we working on today?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                response = main(query=prompt)
                st.markdown(response["answer"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response["answer"]}
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and try again.")
                st.session_state.gemini_api_key = None # Clear the saved key to prompt the user for re-entry
                st.stop() # Stop execution for now
