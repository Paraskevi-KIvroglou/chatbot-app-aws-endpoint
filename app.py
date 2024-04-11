import os
import streamlit as st
import chatbot as demo_model

st.title("Hi, I am Chatbot Philio :woman:")
st.write("I am your hotel booking assistant. Feel free to start chatting with me.")

model = demo_model.load_model()

scrollable_div_style = """
<style>
.scrollable-div {
    height: 200px;  /* Adjust the height as needed */
    overflow-y: auto;  /* Enable vertical scrolling */
    padding: 5px;
    border: 1px solid #ccc;  /* Optional: adds a border around the div */
    border-radius: 5px;  /* Optional: rounds the corners of the border */
}
</style>
"""

def render_chat_history(chat_history):
    #renders chat history
    for message in chat_history:
        if(message["role"]!= "system"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def generate_response(input, memory):
    answer = demo_model.demo_chain(input_text=input, memory= memory)
    if"###Assistant" in answer:
        final_answer = answer.split("###Assistant")
        if len(final_answer) > 1 :
            return final_answer[1]
    return answer

#Application 
#Langchain memory in session cache 
if 'memory' not in st.session_state:
    st.session_state.memory = demo_model.demo_miny_memory()

system_content = "You are a friendly chatbot who always helps the user book a hotel room based on his/her needs.Based on the current social norms you wait for the user's response to your proposals."
#Check if chat history exists in this session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "assistant", "content": "Hello, how can I help you today?"},
    ] #Initialize chat history
    
if 'model' not in st.session_state:
    st.session_state.model = model
    
st.markdown('<div class="scrollable-div">', unsafe_allow_html=True) #add css style to container
render_chat_history(st.session_state.chat_history)

#Input field for chat interface
if input_text := st.chat_input(placeholder="Here you can chat with our hotel booking model."):
    
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append({"role" : "user", "content" : input_text}) #append message to chat history

    with st.spinner("Generating response..."):
        first_answer = generate_response(input_text, st.session_state.memory)
        
        with st.chat_message("assistant"):
            st.markdown(first_answer)
        st.session_state.chat_history.append({"role": "assistant", "content": first_answer})    
st.markdown('</div>', unsafe_allow_html=True)
