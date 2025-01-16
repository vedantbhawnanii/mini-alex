import streamlit as st
import random
import time

from bot import main

st.title("Simple Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What are we working on today?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = main(query=prompt)
        st.markdown(response["answer"])

    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )
