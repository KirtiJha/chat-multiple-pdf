import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.schema import TextEmbeddingParameters
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, SystemMessage

from sentence_transformers import SentenceTransformer

if os.getenv('STREAMLIT_PRODUCTION'):
    gen_ai_key = st.secrets["GENAI_KEY"]
    gen_ai_stream_key = st.secrets["GENAI_CONVERSATION_STREAM_API"]
else:
    load_dotenv()
    gen_ai_key = os.getenv("GENAI_KEY")
    gen_ai_stream_key = os.getenv("GENAI_CONVERSATION_STREAM_API")


# get the text from the pdf files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# split the pdf text to text chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# create a FAISS vector store and store embeddings
def get_vector_store(text_chunks):
    credentials = Credentials(api_key=gen_ai_key)
    client = Client(credentials=credentials)
    embeddings = LangChainEmbeddingsInterface(
        client=client,
        model_id="sentence-transformers/all-minilm-l6-v2",
        parameters=TextEmbeddingParameters(truncate_input_tokens=True),
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


# create llm coversation chain
def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    credentials = Credentials(api_key=gen_ai_key, api_endpoint=gen_ai_stream_key)
    client = Client(credentials=credentials)

    llm = LangChainChatInterface(
        model_id="meta-llama/llama-2-70b-chat",
        client=client,
        parameters=TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE,
            max_new_tokens=100,
            min_new_tokens=10,
            temperature=0.5,
            top_k=50,
            top_p=1,
        ),
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


# handle user input to the chat bot
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    return response['answer']


def main():

    st.title("Chat with multiple PDFs :books:")
    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # accept user input
    if prompt := st.chat_input("Ask a question about your document"):
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # display assistance response in chat message container
        response = handle_user_input(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        # add assistance response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()