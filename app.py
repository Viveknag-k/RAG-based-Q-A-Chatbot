import streamlit as st
import os
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="TalkToPDF",page_icon="📄")
with st.sidebar:
    st.subheader("How to use")
    st.markdown("""<span ><font size=2>1. Start by entering your OpenAI API key.</font></span>""",unsafe_allow_html=True)
    st.markdown("""<span ><font size=2>2. To ask questions related to a document, Upload the document and proceed to ask questions related to the image</font></span>""",unsafe_allow_html=True)
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    uploaded_file = st.file_uploader("Choose a PDF file 📄", accept_multiple_files=False,type="pdf")
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        
    st.divider()
    st.markdown("""<span ><font size=2>Lets Connect!</font></span>""",unsafe_allow_html=True)
    "[Linkedin](https://www.linkedin.com/in/varun-sai-kanuri-089b34226/)" "  \t\t\t"  "[GitHub](https://github.com/varunsai-k)"
    
os.environ['OPENAI_API_KEY']=openai_api_key
openai.api_key = os.environ['OPENAI_API_KEY']

home_title="TalkToPDF"
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=4>Beta</font></span>""",unsafe_allow_html=True)
st.caption("🚀 A streamlit chatbot powered by GPT LLM to talk with PDF 📄")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if uploaded_file is not None:
        if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your 🔑 OpenAI API key to continue.")
                st.stop()
            with open(r"temp_pdf_file.pdf", "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            loader=PyPDFLoader(file_path=r"temp_pdf_file.pdf")
            docs=loader.load()
            llm=OpenAI(temperature=0)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
            splits = text_splitter.split_documents(docs)
            embedding = OpenAIEmbeddings()
            vectordb = FAISS.from_documents(documents=splits,embedding=embedding)
            retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 4, "include_metadata": True})
            template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
            prompts = ChatPromptTemplate.from_template(template)
            rag_chain = RunnableMap({"context": lambda x: retriever.get_relevant_documents(x["question"]),"question": lambda x: x["question"]}) | prompts | llm | StrOutputParser()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response=rag_chain.invoke({"question": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
else:
    
    if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your 🔑 OpenAI API key to continue.")
                st.stop()
            else:
                st.info("Please Upload your pdf 📄 to continue.")
                st.stop()
