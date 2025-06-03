import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
import streamlit as st

# -------------------- App Configuration -------------------- #

st.set_page_config(page_title="LLM Legal Assistant", layout="wide")
st.title("💬 LLM Legal Assistant")

# -------------------- Environment Setup -------------------- #

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("hf_token")
qdrant_token = os.getenv("qdrant_token")

if not hf_token:
    raise EnvironmentError("Hugging Face token (hf_token) is missing in the .env file!")

if not qdrant_token:
    raise EnvironmentError("Qdrant token (qdrant_token) is missing in the .env file!")

# -------------------- Configuration -------------------- #

PDF_PATH = "berlin_services"
COLLECTION_NAME = "berlin_services_alternative"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# -------------------- Clients -------------------- #

# Hugging Face inference for completion
llm_client = InferenceClient(
    provider="cerebras",
    api_key=hf_token
)

# Hugging Face inference for embeddings
embedding_client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token
)

# Qdrant vector DB client
vector_db_client = QdrantClient(
    url="https://4d91633e-ad3c-4c8c-9ba6-66446532f22b.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key=qdrant_token
)

# -------------------- Main Chat Interface -------------------- #

query = st.chat_input("Ask your legal question about Berlin services...")

if query:
    # Get embedding vector for the query
    embedding = embedding_client.feature_extraction(
        query,
        model=EMBEDDING_MODEL,
    )

    # Retrieve top relevant documents
    similar_documents = vector_db_client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=3
    ).points

    # Build context from retrieved documents
    document_blocks = []
    for i, doc in enumerate(similar_documents):
        text = doc.payload.get("text", "")
        block = f"--- Start of Document {i+1} ---\n{text}\n--- End of Document {i+1} ---"
        document_blocks.append(block)

    context = "\n\n".join(document_blocks)

    # Compose full prompt
    full_prompt = (
        f"The following is a user question:\n\n"
        f"\"{query}\"\n\n"
        f"To help answer it, here are some potentially relevant excerpts from official Berlin documents:\n\n"
        f"{context}\n\n"
        f"Based on the documents, please provide a short, clear and informative answer to the user's question regarding Berlin regulations and cite the given documents when possible."
    )

    # Call the LLM
    response = llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content

    # -------------------- Display UI -------------------- #

    st.subheader("🔍 User Question")
    st.markdown(f"> {query}")

    st.subheader("📄 Relevant Documents")
    for i, document in enumerate(similar_documents):
        with st.expander(f"Document {i+1}"):
            st.write(document.payload["text"])

    st.subheader("🤖 LLM Answer")
    st.markdown(answer)

# Navigate with cd command in terminal to current path and the run this app with:
# streamlit run run_chatbot.py
