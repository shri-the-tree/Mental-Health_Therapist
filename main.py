import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create Google Palm LLM model
api_key = ''
llm = GoogleGenerativeAI(google_api_key=api_key, temperature=0.1, model="gemini-1.5-flash-latest")

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def data_ingestion():
    data_directory = "data"
    documents = []

    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Directory '{data_directory}' not found.")

    files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    progress_bar = st.progress(0)
    for i, filename in enumerate(files):
        file_path = os.path.join(data_directory, filename)
        st.write(f"Attempting to load: {file_path}")
        try:
            loader = CSVLoader(file_path=file_path, source_column="Questions", encoding='utf-8')
            file_documents = loader.load()
            st.write(f"Successfully loaded {len(file_documents)} documents from {filename}")
            documents.extend(file_documents)
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
        progress_bar.progress((i + 1) / len(files))

    if not documents:
        raise ValueError("No documents were successfully loaded.")

    st.write(f"Loaded {len(documents)} documents in total.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    st.write(f"Split into {len(docs)} chunks.")

    return docs

def create_vector_db():
    try:
        docs = data_ingestion()
        with st.spinner('Creating vector database... This may take a while.'):
            vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
            vectordb.save_local(vectordb_file_path)
        return "Knowledgebase created successfully!"
    except Exception as e:
        return f"Error creating knowledgebase: {str(e)}"

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "Answers" section in the source document context without making much changes.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

# Streamlit UI
st.title("Mental Health Q&A 🌱")

btn = st.button("Create Knowledgebase")
if btn:
    result = create_vector_db()
    st.write(result)

question = st.text_input("Question: ")

if question:
    try:
        chain = get_qa_chain()
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    # This part will only run if the script is executed directly, not when imported
    pass