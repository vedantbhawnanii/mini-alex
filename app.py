import streamlit as st
from bot import main, call_chain

st.title("Simple Chat")

# Input field for API Key
api_key = st.text_input("Enter your Google API Key:", type="password")

if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()  # Stop execution until API key is provided

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = main(api_key=api_key)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What are we working on today?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = call_chain(prompt, st.session_state.chain)
        st.markdown(response["answer"])

    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )

