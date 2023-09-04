# -*- coding: utf-8 -*-
"""
 @Author: Zeng Peng
 @Date: 2023-09-03 15:00:14
 @Email: zeng.peng@hotmail.com
"""

import streamlit as st
from dotenv import load_dotenv
from utils import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain
from htmlTemplates import css, bot_template, user_template


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        return 
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your courses", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("同你的教材对话 :books:")
    user_question = st.text_input("现在你可以跟你的教材对话:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("你的课程资料")
        pdf_docs = st.file_uploader("上传你的课程资料，并点击‘处理’按钮", accept_multiple_files=True)
        if st.button("处理"):
            with st.spinner("处理中，请稍等..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        

        
if __name__ == '__main__':
    main()