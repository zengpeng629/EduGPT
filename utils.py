# -*- coding: utf-8 -*-
"""
 @Author: Zeng Peng
 @Date: 2023-09-03 15:29:37
 @Email: zeng.peng@hotmail.com
"""

import pdfplumber
import qdrant_client
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Qdrant
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = pdfplumber.open(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_qdrant_collection():
    os.environ['QDRANT_COLLECTION_NAME'] = "Books"
    client = qdrant_client.QdrantClient(
        location=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    # collection_config = qdrant_client.http.models.VectorParams(
    #         size=1536, # 768 for instructor-xl, 1536 for OpenAI
    #         distance=qdrant_client.http.models.Distance.COSINE
    #     )

    # client.recreate_collection(
    #     collection_name=os.getenv("QDRANT_COLLECTION_NAME."),
    #     vectors_config=collection_config
    # )

    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings
    )

    return vectorstore

def get_vectorstore(text_chunks=None):
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = create_qdrant_collection()

    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    converation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return converation_chain

