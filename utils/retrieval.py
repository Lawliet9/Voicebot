"""
retrieval.py
-------------
Handles intelligent document retrieval for the RAG pipeline.

Steps:
1️⃣ Use the detected topic to narrow down the subset of documents.
2️⃣ Compute cosine similarity between query and document embeddings.
3️⃣ Return top-k most relevant documents for generation.
"""

import numpy as np
import re

# ------------------------------------------------------
# Topic-Aware Document Retrieval
# ------------------------------------------------------

def retrieve_context_with_topic(
    query_en: str,
    detected_topic: str,
    vectorstore,
    emb,
    top_k: int = 3
):
    """
    Retrieve top-k relevant documents using both detected topic and semantic similarity.

    Args:
        query_en (str): English-translated user query
        detected_topic (str): Topic/category detected by Gemini
        vectorstore: Loaded FAISS vectorstore
        emb: HuggingFaceEmbeddings object
        top_k (int): Number of documents to return

    Returns:
        List of tuples -> [(Document, similarity_score), ...]
    """

    # # ---- Step 1: Normalize topic text ----
    # ---- Step 1: Normalize topic text ----
    detected_topic_norm = re.sub(r'[^a-zA-Z0-9\s]', '', detected_topic or "").lower().strip()

    # ---- Step 2: Fuzzy filter by topic (partial match in category or scheme name) ----
    all_docs = list(vectorstore.docstore._dict.values())
    filtered_docs = []
    for d in all_docs:
        meta = d.metadata or {}
        cat = str(meta.get("category", "")).lower()
        scheme = str(meta.get("scheme_name", "")).lower()

        if (
            detected_topic_norm in cat
            or detected_topic_norm in scheme
            or cat in detected_topic_norm
            or scheme in detected_topic_norm
        ):
            filtered_docs.append(d)

    # ---- Step 3: Fallback if nothing matches topic ----
    if not filtered_docs:
        print(f"⚠️ No docs found for topic '{detected_topic}', using all documents.")
        filtered_docs = all_docs

    print(len(all_docs))
    print("total filtered docs ",len(filtered_docs))
    # ---- Step 4: Compute embeddings ----
    query_vec = emb.embed_query(query_en + " " + detected_topic)
    #filtered_docs = all_docs
    doc_texts = [
    f"{d.metadata.get('category', '')} {d.metadata.get('scheme_name', '')} {d.page_content}"
    for d in filtered_docs
    ]
    doc_vecs = emb.embed_documents(doc_texts)

    # ---- Step 5: Compute cosine similarity ----
    sims = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )


    # ---- Step 6: Rank and return ----
    ranked = sorted(zip(filtered_docs, sims), key=lambda x: x[1], reverse=True)
    top_results = ranked[:top_k]

    print(f"✅ Retrieved top-{len(top_results)} documents for topic '{detected_topic}'.")
    return top_results


# ------------------------------------------------------
# Confidence Scoring Helper
# ------------------------------------------------------

# def compute_confidence(similarities, threshold: float = 0.5):
#     """
#     Compute normalized confidence from cosine similarity values.
#     Returns 0.0–1.0 confidence score based on highest similarity.
#     """
#     if not similarities:
#         return 0.0

#     top_score = float(similarities[0])
#     # Clamp similarity (cosine between -1 and 1)
#     conf = max(0.0, min(1.0, (top_score - threshold) / (1 - threshold)))
#     return round(conf, 3)


# ------------------------------------------------------
# CLI Test
# ------------------------------------------------------

if __name__ == "__main__":
    from vectorstore import load_vectorstore

    #vs, emb = load_vectorstore("vectorstore/schemes_faiss_index")
    vs, emb = load_vectorstore("vectorstore_new.faiss")
    query_en = "Tell me about the Kisan Credit Card"
    detected_topic = "Kisan Credit Card"

    results = retrieve_context_with_topic(query_en, detected_topic, vs, emb, top_k=3)
   # sims = [r[1] for r in results]
    #conf = compute_confidence(sims)

    # print("\n🔎 Top Results:")
    # for doc, score in results:
    #     print(f"→ {doc.metadata.get('scheme_name')} ({score:.3f})")
    # print(f"\nConfidence Score: {conf}")
    print(results)
