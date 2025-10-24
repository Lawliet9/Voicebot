import streamlit as st
from streamlit_mic_recorder import mic_recorder
from utils.rag_pipeline import run_rag_pipeline
import os

# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="🎙️ Multilingual VoiceBot", layout="centered")
st.title("🎙️ Multilingual VoiceBot (RAG + Memory + Mic Input)")
st.markdown("Ask questions about Indian Government Schemes in your preferred language!")

# ------------------------------------------------------------
# Initialize Session State Variables
# ------------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = []

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

MAX_MEMORY_TURNS = 3

# ------------------------------------------------------------
# Sidebar: Settings
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    lang_map = {
        "Hindi": "hi",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "English": "en",
    }

    user_lang_display = st.selectbox("Select your preferred language:", list(lang_map.keys()))
    user_lang = lang_map[user_lang_display]

    st.markdown("---")
    st.info("🎤 You can either record your voice or upload an audio file.")

# ------------------------------------------------------------
# Audio Input Section
# ------------------------------------------------------------
input_mode = st.radio("Choose input mode:", ["🎙️ Record via Mic", "📁 Upload Audio File"])

# 1️⃣ MIC Recording Mode
if input_mode == "🎙️ Record via Mic":
    st.write("Press below to record your question 👇")
    audio_data = mic_recorder(
        start_prompt="🎤 Record",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key="recorder",
    )

    if audio_data:
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            st.session_state.audio_data = audio_data["bytes"]
        else:
            st.session_state.audio_data = audio_data
        st.audio(st.session_state.audio_data, format="audio/wav")
        st.success("🎤 Audio recorded successfully!")

# 2️⃣ File Upload Mode
elif input_mode == "📁 Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        st.session_state.audio_data = uploaded_file.read()
        st.audio(st.session_state.audio_data, format="audio/wav")
        st.success("📁 Audio file uploaded successfully!")

# ------------------------------------------------------------
# Process Query Button
# ------------------------------------------------------------
if st.session_state.audio_data is not None:
    if st.button("🚀 Process Query"):
        with st.spinner("Processing your audio query... ⏳"):
            try:
                result = run_rag_pipeline(st.session_state.audio_data, user_lang=user_lang)
                st.session_state.last_result = result
                st.session_state.memory.append({
                    "user": result.get("transcript", ""),
                    "bot": result.get("bot_response", "")
                })
                st.session_state.memory = st.session_state.memory[-MAX_MEMORY_TURNS:]
                st.success("✅ Query processed successfully!")
            except Exception as e:
                st.error(f"❌ Error running pipeline: {e}")

else:
    st.info("🎧 Please record or upload an audio query first.")

# ------------------------------------------------------------
# Display Results (after pipeline completes)
# ------------------------------------------------------------
if st.session_state.last_result:
    result = st.session_state.last_result

    st.subheader("🗣️ Transcript")
    st.write(result.get("transcript", ""))

    st.subheader("🌐 English Translation")
    st.write(result.get("query_en", ""))

    st.subheader("🧭 Detected Topic")
    st.write(result.get("detected_topic", ""))

    st.subheader("📚 Retrieved Context (Top 3)")
    for doc in result.get("retrieved_docs", []):
        st.markdown(
            f"**{doc['scheme']}** — *{doc['category']}*  \n"
            f"🔗 [{doc['url']}]({doc['url']})  \n"
            f"Score: {doc['score']}"
        )

    st.subheader("💬 Bot Response")
    st.write(result.get("bot_response", ""))

    st.subheader("🎧 Listen to Response")
    if result.get("tts_path") and os.path.exists(result["tts_path"]):
        st.audio(result["tts_path"], format="audio/wav")
    else:
        st.info("No audio response available.")

# ------------------------------------------------------------
# Conversation Memory
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🧠 Conversation Memory (Last 3 Turns)")

if st.session_state.memory:
    for i, turn in enumerate(st.session_state.memory[-MAX_MEMORY_TURNS:], 1):
        st.markdown(f"**Turn {i}:**")
        st.markdown(f"👤 User: {turn['user']}")
        st.markdown(f"🤖 Bot: {turn['bot']}")
        st.markdown("---")
else:
    st.info("Start by speaking or uploading your first query!")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using AI4Bharat, Gemini & LangChain</p>",
    unsafe_allow_html=True,
)
