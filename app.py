import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if the API key is loaded correctly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key not found in environment variables. Please check 'key.env'.")

# Configure the API key
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Instructions:
    You are a knowledgeable assistant focused on providing safety guidelines for coastal areas during cyclones. Your goal is to generate personalized, clear, and actionable advice based on the specific details provided about the user's infrastructure, proximity to the cyclone pathway,cyclone speed and proximity to the nearest shelter, person's location.
    
    Please:
    - Carefully analyze the provided context from the PDF.
    - Offer tailored guidance that addresses the user's unique situation.
    - calculating nearest shelter by location of the person(lat,lon) and shelter cordinates(lat,lon)
    - calculating Proximity to cyclone by location of the person(lat,lon) and Predicted Cyclone Cordinates(lat,lon).
    - Ensure that your advice is practical and directly applicable.
    - If information is missing or unclear, use logical assumptions based on the context to provide the best possible recommendations.
    - Be concise but thorough, offering detailed steps when necessary to enhance safety and preparedness.
    
    Context:\n{context}\n
    Question: \n{question}\n

    Personalized Guideline:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("🌪️Generate Tailored Guideline-Cyclone")

    # Input fields for user information
    infrastructure = st.text_input("Infrastructure")
    location = st.text_input("Location Cordinate(lat,lon)")
    cyclone_predicted_cordinates = st.text_input("Predicted Cyclone Cordinates(lat,lon)")
    cyclone_speed = st.text_input("Cyclone Speed in Knots")

    if infrastructure and location and cyclone_predicted_cordinates and cyclone_speed:
        user_question = f"{infrastructure} Infrastructure, location of the person(lat,lon) is {location}, Cyclone Speed in knot is {cyclone_speed}, Predicted Cyclone Cordinates(lat,lon) is {cyclone_predicted_cordinates}.Please give guideline what will be best in this context.Give Precise instruction by calculating Proximity to cyclone by location of the person(lat,lon) and Predicted Cyclone Cordinates(lat,lon).Also give the location of the nearest shelter by calculating location of the person(lat,lon) and shelter cordinates(lat,lon)(from the text chunk given).DOnt give Proximity to cyclone and Proximity to shelter though(only use this to generate the guideline). also give the helpline number in last:333."
        user_input(user_question)

    with st.sidebar:
        st.title("Documents:")

        # Only use "guideline1.pdf" as the default PDF
        default_pdf = "guideline1.pdf"
        default_pdf_doc = open(default_pdf, "rb")
        st.write("Default PDF:")
        st.write(default_pdf)

        # Automatically process the PDF when the app starts
        raw_text = get_pdf_text([default_pdf_doc])
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

if __name__ == "__main__":
    main()