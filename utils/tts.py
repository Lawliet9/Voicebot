"""
tts.py
------
Handles Text-to-Speech (TTS) for the multilingual voicebot.
Uses Google's gTTS to synthesize vernacular speech output.
"""

import os
from gtts import gTTS
import tempfile


# ----------------------------------------------------------
# Text-to-Speech Function
# ----------------------------------------------------------

def tts_gtts(
    text: str,
    lang_code: str,
    output_path: str | None = None
) -> str | None:
    """
    Convert text into speech and save as an audio file.

    Args:
        text (str): Text to convert to speech.
        lang_code (str): Language code (hi, mr, ta, te, bn, gu, ml, or en).
        output_path (str, optional): File path to save output audio. 
            If None, creates a temporary file.

    Returns:
        str | None: Path to generated audio file (None if failed).
    """
    if not text or not text.strip():
        print("⚠️ No text provided for TTS.")
        return None

    try:
        print(f"🎤 Generating speech in language: {lang_code}")
        tts = gTTS(text=text, lang=lang_code)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            output_path = tmp.name

        tts.save(output_path)
        print(f"✅ TTS audio saved at: {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ TTS generation failed: {e}")
        return None


# ----------------------------------------------------------
# Optional: Playback Helper (for Jupyter/Streamlit testing)
# ----------------------------------------------------------

def play_audio(filepath: str):
    """
    Plays audio inline (works in Jupyter and Streamlit).
    """
    from IPython.display import Audio, display
    if os.path.exists(filepath):
        print("in")
        display(Audio(filepath))
    else:
        print("⚠️ Audio file not found:", filepath)


# ----------------------------------------------------------
# CLI Test
# ----------------------------------------------------------

if __name__ == "__main__":
    sample_text = "किसान क्रेडिट कार्ड योजना किसानों को वित्तीय सहायता प्रदान करती है।"
    lang = "hi"
    audio_path = tts_gtts(sample_text, lang_code=lang, output_path = './utils/test.wav')
    print(audio_path)
    if audio_path:
        print("here")
        play_audio(audio_path)
