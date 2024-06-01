import main
from main import rag_chain
import streamlit as st

# Set the page configuration for Streamlit
st.set_page_config(page_title="Lucy Skywalker")

# Sidebar content
with st.sidebar:
    st.title('About Lucy Skywalker')

# Function for generating LLM response
def generate_response(input):
    result = rag_chain.invoke(input)
    return result

# Initialize the session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about Lucy Skywalker!"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input("Ask for a game recommendation..."):
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer from mystery stuff..."):
                response = generate_response(input)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
