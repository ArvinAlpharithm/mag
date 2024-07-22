import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM
from langchain.chains.question_answering import load_qa_chain
import os
import pickle
import base64
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@st.cache_resource
def initialize_model():
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1000,
        "min_new_tokens": 1
    }
    return WatsonxLLM(
        model_id="mistralai/mixtral-8x7b-instruct-v01",
        apikey="3KCJ_Pf60xnWETCVpQ2qSTW5rVN1z0GyHv5nUh0DoyM6",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="aff3b33e-7baa-484f-8ce5-540f75416272",
        params=parameters,
    )

def get_pdf_display(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    return pdf_display

def split_text_into_sentences(text):
    # Split text into sentences using regex for better accuracy
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def main():
    st.title("Alpharithm Magma POC")

    llm = initialize_model()

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_path = uploaded_file.name

        # Save the uploaded file temporarily
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_reader = PdfReader(pdf_path)
        total_pages = len(pdf_reader.pages)

        if 'page_num' not in st.session_state:
            st.session_state.page_num = 0

        # Question input section
        query = st.text_input("**Ask questions about One Health Extracover Policy:**")

        answer_placeholder = st.empty()
        with answer_placeholder.container():
            st.markdown("**Answer:**")

        if query:
            if 'VectorStore' not in st.session_state:
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(text=text)

                # Embeddings
                store_name = os.path.splitext(os.path.basename(pdf_path))[0]

                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        st.session_state.VectorStore = pickle.load(f)
                else:
                    # Using HuggingFace embeddings instead of OpenAI
                    embeddings = HuggingFaceEmbeddings()
                    st.session_state.VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(st.session_state.VectorStore, f)

            docs = st.session_state.VectorStore.similarity_search(query=query, k=3)

            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Add system message to guide the model
            system_message = (
                "This is an insurance policy document, so always explain the point in detail in the simplest way possible."
            )
            result = chain.invoke({"input_documents": docs, "question": query, "system_message": system_message})
            response = result["output_text"]

            # Determine the one-word summary of the response
            summary = response.split()[0] if response else "No"

            # Check if the response already contains bullet points
            if any(line.strip().startswith("- ") for line in response.splitlines()):
                # Response already has bullet points, display as is
                formatted_response = response
            else:
                # Split the response into sentences
                sentences = split_text_into_sentences(response)
                # Prepare the response with bullet points
                formatted_response = "\n".join([f"- {sentence}" for sentence in sentences])

            answer_placeholder.empty()
            with answer_placeholder.container():
                st.markdown(formatted_response, unsafe_allow_html=True)

        # Display the PDF in an iframe
        st.write("### PDF Preview:")
        pdf_display = get_pdf_display(pdf_path)
        st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
