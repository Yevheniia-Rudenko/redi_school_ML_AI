import os
import uuid
from dotenv import load_dotenv
import PyPDF2
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

# -------------------- Configuration -------------------- #

# Directory containing PDF files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of the current script file
PDF_PATH = os.path.join(SCRIPT_DIR, "berlin_services_pdfs")
# Name of the Qdrant collection
COLLECTION_NAME = "berlin_services_alternative"

# Embedding model to use from Hugging Face
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Load environment variables
load_dotenv()
hf_token = os.getenv("hf_token")
qdrant_token = os.getenv("qdrant_token")

# -------------------- Clients Setup -------------------- #

# Hugging Face Inference API client
embedding_client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token
)

# Qdrant vector database client
vector_db_client = QdrantClient(
    url="https://4d91633e-ad3c-4c8c-9ba6-66446532f22b.eu-central-1-0.aws.cloud.qdrant.io:6333", # your own link to free aws machine registered in qdrant
    api_key=qdrant_token
)

# -------------------- Helper Functions -------------------- #

def load_pdfs_from_folder(folder_path: str):
    """
    Load all PDF file paths from the given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        List[str]: List of full paths to PDF files.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

def convert_pdf_to_text(pdf_file: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_file (str): Path to the PDF file.

    Returns:
        str: Extracted text or empty string on failure.
    """
    try:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_file}: {e}")
        return ""

def recreate_collection():
    """
    Recreate the Qdrant collection, deleting it first if it exists.
    """
    try:
        vector_db_client.delete_collection(COLLECTION_NAME)
        vector_db_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            timeout=10
        )
        print(f"[INFO] Collection '{COLLECTION_NAME}' recreated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to recreate collection: {e}")

# -------------------- Main Logic -------------------- #

def main():
    recreate_collection()
    pdf_files = load_pdfs_from_folder(PDF_PATH)

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        text = convert_pdf_to_text(pdf_file)
        if not text.strip():
            print(f"[WARNING] Skipping empty PDF: {os.path.basename(pdf_file)}")
            continue

        try:
            embedding = embedding_client.feature_extraction(
                text,
                model=EMBEDDING_MODEL,
            )

            if len(embedding) != 1024:
                print(f"[WARNING] Unexpected embedding size for {os.path.basename(pdf_file)}: {len(embedding)}")
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(pdf_file)))

            vector_db_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={"text": text},
                    )
                ]
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {os.path.basename(pdf_file)}: {e}")

if __name__ == "__main__":
    main()