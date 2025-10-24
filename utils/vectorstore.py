"""
vectorstore.py
----------------
Handles corpus preparation, embedding generation, and FAISS vector store creation
for the AI4Bharat VoiceBot RAG system.
"""

import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pickle

# ----------------------------
# Configurable Parameters
# ----------------------------

CSV_PATH = "schemes_multilingual.csv"
VECTORSTORE_PATH = "vectorstore_new.faiss"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ----------------------------
# Corpus Preparation
# ----------------------------

def load_and_prepare_documents(csv_path: str = CSV_PATH):
    """
    Loads multilingual scheme data and converts it into a list of LangChain Documents.
    Each language column (English, Hindi, etc.) becomes a separate document entry.
    """
    print("📂 Loading dataset:", csv_path)
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        for lang in ["english", "hindi"]:
            text = str(row[lang]).strip()
            if not text or text.lower() == "nan":
                continue
            documents.append({
                "id": row["id"],
                "scheme_name": row["scheme_name"],
                "category": row["category"],
                "language": lang,
                "content": text,
                "source_url": row["source_url"]
            })

    print(f"✅ Prepared {len(documents)} multilingual documents.")
    return documents


# ----------------------------
# Text Splitting
# ----------------------------

def split_documents(docs):
    """
    Split long documents into smaller overlapping chunks for better embedding representation.
    """
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    #splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "id": doc["id"],
                "scheme_name": doc["scheme_name"],
                "category": doc["category"],
                "language": doc["language"],
                "text": chunk,
                "source_url": doc["source_url"],
                "chunk_id": f"{doc['id']}_{i}"
        })
    return chunks


# ----------------------------
# Embedding Model
# ----------------------------

def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Load the HuggingFace multilingual embedding model.
    """
    print(f"🔢 Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


# ----------------------------
# Vector Store Creation
# ----------------------------

def create_vectorstore(chunks, embeddings, save_path: str = VECTORSTORE_PATH):
    """
    Create a FAISS vector store from documents and save it locally.
    """
    print("🧠 Creating FAISS vector store...")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "id": chunk["id"],
            "scheme_name": chunk["scheme_name"],
            "category": chunk["category"],
            "language": chunk["language"],
            "source_url": chunk["source_url"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    #vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, normalize_L2=True)
    print(f"💾 Saving vector store to: {save_path}")
    vectorstore.save_local(save_path)
    print("✅ Vector store saved successfully.")
    return vectorstore


# ----------------------------
# Load Existing Vectorstore
# ----------------------------

def load_vectorstore(save_path: str = VECTORSTORE_PATH, model_name: str = EMBEDDING_MODEL):
    """
    Load an existing FAISS vector store from disk.
    """
    print(f"📦 Loading vector store from {save_path}...")
    embeddings = get_embeddings(model_name)
    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded successfully.")
    return vectorstore, embeddings


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    docs = load_and_prepare_documents(CSV_PATH)
    chunks = split_documents(docs)
    emb = get_embeddings(EMBEDDING_MODEL)
    vs = create_vectorstore(chunks, emb, VECTORSTORE_PATH)
