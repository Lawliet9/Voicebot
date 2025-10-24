"""
rag_pipeline.py
---------------
Full RAG pipeline orchestration for the multilingual AI VoiceBot.

Flow:
1️⃣ Speech-to-Text (AI4Bharat)
2️⃣ Translation → English (Gemini)
3️⃣ Topic Extraction (Gemini)
4️⃣ Document Retrieval (FAISS + Embeddings)
5️⃣ Response Generation (Gemini)
6️⃣ Text-to-Speech (gTTS)
"""

import os
from utils.stt import transcribe_ai4bharat
from utils.translation import translate_to_english, extract_topic_from_query
from utils.retrieval import retrieve_context_with_topic
from utils.vectorstore import load_vectorstore
from utils.tts import tts_gtts

# from stt import transcribe_ai4bharat
# from translation import translate_to_english, extract_topic_from_query
# from retrieval import retrieve_context_with_topic
# from vectorstore import load_vectorstore
# from tts import tts_gtts
import google.generativeai as genai

# ---------------------------------------------------------------------
# Gemini Setup
# ---------------------------------------------------------------------

#genai.configure(api_key="AIzaSyCoI_9gL_iLoNcJMUz6g9Z-jxFVbMG98Bg")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------------------------------------------------
# RAG Prompt Template
# ---------------------------------------------------------------------

# RAG_PROMPT_TEMPLATE = """
# You are a multilingual assistant that answers user queries using only the provided English context.
# If the query was in another language, think in English, find the answer from the context,
# and then respond back in the user's original language ({user_lang}).

# Note - The context will talk about 2-3 different schemes ,
# you have to look at the query and see which scheme is mostly matching the user query and anwser accordingly

# Context (English):
# {context}

# User Query (Original: {user_lang}):
# {query}

# Answer the question concisely in the user's original language.
# If no relevant information is found, say: "Mujhe is vishay mein jaankari nahi hai."
# Answer:
# """

RAG_PROMPT_TEMPLATE = """
You are a multilingual assistant that answers user queries using only the provided English context.
If the query was in another language, think in English, find the answer from the context,
and then respond back in the user's original language ({user_lang}).

Note - The context will talk about 2-3 different schemes,
you have to look at the query and see what kind of response to generate by looking at context.
It can be general response summing the context or a specific response abput a scheme

Note - You also have a section of user and bots recent conversation history.
For first turn it will be empty so answer via query and context.
But if conversations exists then also keep in mind the history as an additional context and answer accordingly.

Context (English):
{context}

User Query (Original: {user_lang}):
{query}

Previous_history:
{history}

Answer the question concisely in the user's original language.
If no relevant information is found, say: "Mujhe is vishay mein jaankari nahi hai."
Answer:
"""



RAG_PROMPT_TEMPLATE_NO_INFO = """
You are required to anwser for cases where RAG confidence was low

Note - You have to convert a static english statement to regional language by using the language code provided if there is no conversation history provided.
But if some conversation history exist then use that and try to use that as context and anwser user's query.  

User Query (Original: {user_lang}):
{query}

Previous_history:
{history}

static english statement: I dont have any information regarding this query.

Convert the above statement to this langguage  :{user_lang} if Previous_history is empty else use history to anwser query again in this language ::{user_lang}

Answer:
"""

# ---------------------------------------------------------------------
# Main RAG Routine
# ---------------------------------------------------------------------

def run_rag_pipeline(
    audio_file,
    user_lang="hi",
    decoding="rnnt",
    top_k=3,
    confidence_threshold=0.4,
    memory=None
):
    """
    Runs the complete RAG pipeline:
    1. Transcribe → 2. Translate → 3. Extract Topic → 4. Retrieve → 5. Generate → 6. TTS

    Returns:
        dict -> {
            "transcript": str,
            "query_en": str,
            "detected_topic": str,
            "retrieved_docs": list,
            "confidence": float,
            "bot_response": str,
            "tts_path": str
        }
    """
    # ----------------- Load Vectorstore -----------------
    vs, emb = load_vectorstore("vectorstore_new.faiss")

    # ----------------- PHASE 1: STT -----------------
    print("🎧 Running Speech-to-Text...")
    stt_out = transcribe_ai4bharat(audio_file if isinstance(audio_file, (bytes, bytearray)) else audio_file, lang_code=user_lang, decoding=decoding)
    transcript = stt_out.get("transcript", "").strip()
    detected_lang = stt_out.get("detected_language", user_lang)
    print(f"🗣️ Transcript: {transcript}")

    if not transcript:
        print("⚠️ No valid transcript found. Skipping to safe return.")
        return {
            "error": "No transcript detected.",
            "transcript": "",
            "query_en": "",
            "detected_topic": "",
            "retrieved_docs": [],
            "confidence": 0.0,
            "bot_response": "मुझे आपकी आवाज़ समझ में नहीं आई, कृपया फिर से बोलिए।",
            "tts_path": None,
        }


    # ----------------- PHASE 2: Translation -----------------
    print("🌐 Translating to English...")
    query_en = translate_to_english(transcript)
    print(f"✅ English Query: {query_en}")

    # ----------------- PHASE 3: Topic Extraction -----------------
    print("🧭 Extracting topic...")
    from pandas import read_csv
    df = read_csv("schemes_multilingual.csv")
    topics = list(df["category"].dropna().unique())
    detected_topic = extract_topic_from_query(query_en, topics)
    print(f"🧩 Detected Topic: {detected_topic}")

    # ----------------- PHASE 4: Retrieval -----------------
    print("📚 Retrieving relevant documents...")
    retrieved = retrieve_context_with_topic(query_en, detected_topic, vs, emb, top_k=top_k)
    print(retrieved)
    # sims = [r[1] for r in retrieved]
    # confidence = compute_confidence(sims)
    confidence = retrieved[0][1]
    print(f"✅ Confidence Score: {confidence}")

    # Build context for generation
    context_parts = []
    for doc, score in retrieved:
        meta = doc.metadata
        scheme_name = meta.get("scheme_name", "")
        category = meta.get("category", "")
        url = meta.get("source_url", "")
        content = doc.page_content.strip()
        paragraph = (
            f"The scheme **{scheme_name}** belongs to the **{category}** category. "
            f"It focuses on {content}. For more information, visit {url}."
        )
        context_parts.append(paragraph)
    context_text = "\n\n".join(context_parts)

    # ----------------- PHASE 5: Generation -----------------

    # 🧠 Build history text from Streamlit memory (if available)
    history_text = ""
    if memory:
        history_text = "\n".join([f"User: {t['user']}\nBot: {t['bot']}" for t in memory])

    print("🤖 Generating final response...")
    if confidence < confidence_threshold:
        prompt = RAG_PROMPT_TEMPLATE_NO_INFO.format(user_lang = user_lang, query=query_en, history=history_text)

    else:
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_text, user_lang=user_lang, query=query_en, history=history_text
        )

    response = model.generate_content(prompt)
    bot_response = response.text.strip()
    print("💬 Bot Response:", bot_response)

    # ----------------- PHASE 6: TTS -----------------
    print("🔊 Generating speech output...")
    tts_path = tts_gtts(bot_response, lang_code=user_lang,output_path = './utils/user_output.wav')
    if tts_path:
        print(f"🎵 Audio response saved at: {tts_path}")

    # ----------------- RETURN -----------------

    
    return {
        "transcript": transcript,
        "query_en": query_en,
        "detected_topic": detected_topic,
        "retrieved_docs": [
            {
                "scheme": doc.metadata.get("scheme_name", ""),
                "category": doc.metadata.get("category", ""),
                "url": doc.metadata.get("source_url", ""),
                "content":doc.metadata.get("page_content", ""),
                "score": round(score, 3)
            }
            for doc, score in retrieved
        ],
        "confidence": confidence,
        "bot_response": bot_response,
        "tts_path": tts_path
    }


# ---------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import os

    audio_path = os.path.join("utils", "user_query.wav")  # Example audio
    if not os.path.exists(audio_path):
        print("⚠️ Please place 'user_query.wav' inside utils/")
    else:
        result = run_rag_pipeline(audio_path, user_lang="hi")
        print("\n=== PIPELINE RESULT ===")
        for k, v in result.items():
            print(f"{k}: {v}\n")
