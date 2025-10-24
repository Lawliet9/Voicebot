import gradio as gr
import os
from utils.rag_pipeline import run_rag_pipeline

# ------------------------------------------------------------
# Global Memory
# ------------------------------------------------------------
MAX_TURNS = 3
session_memory = []


# ------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------
def voicebot(audio_file, lang_display, state):
    """
    Main callable: handles voice -> RAG -> response.
    Returns detailed output for UI.
    """
    try:
        # 🔸 Ensure consistent return shape (8 outputs)
        # [state, transcript, english, topic, context, tts_path, bot_response, chat_history]
        if audio_file is None:
            return (
                state,
                "⚠️ Please record your question!",
                "",
                "",
                "",
                None,
                "",
                "",
            )

        # Language mapping
        lang_map = {
            "Hindi": "hi",
            "Marathi": "mr",
            "Tamil": "ta",
            "Telugu": "te",
            "Bengali": "bn",
            "English": "en",
        }
        user_lang = lang_map.get(lang_display, "hi")

        # Run RAG pipeline
        result = run_rag_pipeline(audio_file, user_lang=user_lang, memory=state)

        # Extract outputs
        transcript = result.get("transcript", "")
        query_en = result.get("query_en", "")
        detected_topic = result.get("detected_topic", "")
        retrieved_docs = result.get("retrieved_docs", [])
        bot_response = result.get("bot_response", "")
        tts_path = result.get("tts_path", None)

        # Build retrieved context section
        if retrieved_docs:
            retrieved_text = "\n\n".join(
                [
                    f"**{d['scheme']}** — *{d['category']}*\n🔗 [{d['url']}]({d['url']})\nScore: {d['score']}"
                    for d in retrieved_docs
                ]
            )
        else:
            retrieved_text = "No relevant context found."

        # Update conversation memory
        state.append({"user": transcript, "bot": bot_response})
        if len(state) > MAX_TURNS:
            state = state[-MAX_TURNS:]

        # Build conversation history
        chat_history = "\n".join(
            [f"👤 User: {t['user']}\n🤖 Bot: {t['bot']}" for t in state]
        )

        # ✅ Return all 8 outputs
        return (
            state,
            transcript,
            query_en,
            detected_topic,
            retrieved_text,
            tts_path,
            bot_response,
            chat_history,
        )

    except Exception as e:
        print("❌ Error:", e)
        # Return 8 placeholders on error
        return (
            state,
            "",
            "",
            "",
            "",
            None,
            f"Error: {e}",
            "",
        )


# ------------------------------------------------------------
# Reset Conversation
# ------------------------------------------------------------
def reset_conversation():
    return [], "", "", "", "", None, "Conversation reset.", ""


# ------------------------------------------------------------
# Build Gradio UI
# ------------------------------------------------------------
def build_interface():
    with gr.Blocks(title="🎙️ Multilingual VoiceBot") as demo:
        gr.Markdown(
            """
            # 🎙️ Multilingual VoiceBot (RAG + AI4Bharat + Gemini)
            Ask about Indian Government Schemes in your preferred language!  
            _You can record up to 3 turns per conversation._
            """
        )

        # Language and reset controls
        with gr.Row():
            lang = gr.Dropdown(
                ["Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "English"],
                label="🌐 Choose your language",
                value="Hindi",
            )
            reset_btn = gr.Button("🔄 Reset Conversation")

        # Audio input
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="🎧 Record or Upload Audio Query",
        )

        # Outputs (structured layout)
        with gr.Row():
            transcript_box = gr.Textbox(label="🗣️ Transcript (Speech to Text)", interactive=False)
            english_box = gr.Textbox(label="🌐 English Translation", interactive=False)

        with gr.Row():
            topic_box = gr.Textbox(label="🧭 Detected Topic", interactive=False)

        with gr.Row():
            context_box = gr.Markdown(label="📚 Retrieved Context (Top 3)")

        bot_box = gr.Textbox(label="💬 Bot Response", interactive=False)
        audio_output = gr.Audio(label="🔊 Bot Audio Response")
        chat_history = gr.Textbox(label="🧠 Conversation History (Last 3 Turns)", interactive=False, lines=8)

        # Hidden memory state
        state = gr.State([])

        # Interaction binding
        audio_input.change(
            fn=voicebot,
            inputs=[audio_input, lang, state],
            outputs=[
                state,
                transcript_box,
                english_box,
                topic_box,
                context_box,
                audio_output,
                bot_box,
                chat_history,
            ],
        )

        reset_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=[
                state,
                transcript_box,
                english_box,
                topic_box,
                context_box,
                audio_output,
                bot_box,
                chat_history,
            ],
        )

    return demo


# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
if __name__ == "__main__":
    app = build_interface()
    app.launch()
