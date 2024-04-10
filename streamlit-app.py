import streamlit as st
from AI_model import get_model
from conv_chain import ConvChain

st.title("NHS Document Chat")

# call the model
llm_model = get_model("GPT")  # replace with "GPT"  or "VertexAI"
cc = ConvChain(llm_model)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = cc.get_response(query)
        address = [item.metadata['source'] for item in response['source_documents']]
        full_response += response['answer']
        if len(address) == 1:
            full_response += " \n I recommend taking a look at the following page: "
        elif len(address) > 1:
            full_response += " \n I recommend taking a look at the following pages: "
        for item in set(address):
            full_response += " \n " + item
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
# Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
