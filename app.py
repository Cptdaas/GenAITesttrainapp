import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint # Updated import
from langchain_core.prompts import PromptTemplate # Updated import
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set the API token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    st.error("Hugging Face API token not found in .env file.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Initialize the LLM (using the recommended HuggingFaceEndpoint)
llm = HuggingFaceEndpoint( # Updated class
    repo_id="google/flan-t5-large",
    temperature=0.5, 
    max_new_tokens=1000
)

# Define prompt template
prompt = PromptTemplate(
    input_variables=["query"],
    template="Answer the following question in a clear and concise way:\n\nQuestion: {query}\n\nAnswer:"
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit UI
st.title("💬 LangChain + Hugging Face Chatbot")
st.write("Ask any question below:")

query = st.text_input("Your question")

if query:
    with st.spinner("Thinking..."):
        response = llm_chain.run(query=query)
        st.success("Answer:")
        st.write(response)

