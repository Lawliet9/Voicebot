"""
stt.py
------
Handles Speech-to-Text (STT) conversion for the multilingual voicebot using
AI4Bharat's Indic Conformer 600M Multilingual model.
"""

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
import tempfile
import os
import io



# ----------------------------
# Model Setup
# ----------------------------
from huggingface_hub import login

# Load the token from environment
hf_token = os.getenv("hf_token")

# Log in securely
if hf_token:
    login(token=hf_token)
    print("✅ Successfully logged into Hugging Face Hub!")
else:
    print("⚠️ HF_TOKEN not found. Please set it as an environment variable.")


MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"

print(f"🔊 Loading AI4Bharat STT model: {MODEL_NAME}")
# trust_remote_code is needed for custom forward pass
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)


# ----------------------------
# Helper Functions
# ----------------------------

def load_audio(audio_input):
    """
    Load audio from bytes, Streamlit upload, or file path.
    Converts any playable format (wav/mp3/m4a/webm) → 16kHz mono waveform.
    """
    import io
    import torch
    import torchaudio
    from pydub import AudioSegment

    try:
        # Case 1: audio_input is raw bytes (mic or upload)
        if isinstance(audio_input, (bytes, bytearray)):
            audio_buffer = io.BytesIO(audio_input)
            segment = AudioSegment.from_file(audio_buffer)
        # Case 2: Streamlit UploadedFile
        elif hasattr(audio_input, "read"):
            segment = AudioSegment.from_file(audio_input)
        # Case 3: file path
        else:
            segment = AudioSegment.from_file(audio_input)

        # Convert to mono and resample
        segment = segment.set_channels(1)
        segment = segment.set_frame_rate(16000)

        # Convert to PyTorch tensor
        samples = torch.tensor(segment.get_array_of_samples(), dtype=torch.float32)
        waveform = samples.unsqueeze(0) / 32768.0  # normalize to [-1, 1]
        sr = 16000

        return waveform, sr

    except Exception as e:
        raise RuntimeError(f"Error loading audio input: {e}")



# ----------------------------
# Transcription Function
# ----------------------------

def transcribe_ai4bharat(audio_input, lang_code="hi", decoding="rnnt"):
    """
    Transcribe an audio input (file path or bytes) using AI4Bharat model.
    """
    try:
        print("🎧 Loading audio...")
        waveform, sr = load_audio(audio_input)
        waveform = waveform.to(DEVICE)

        print(f"🗣️ Running ASR inference in '{lang_code}' using {decoding.upper()} decoding...")
        transcript = model(waveform, lang_code, decoding)

        result = {
            "transcript": transcript.strip(),
            "detected_language": lang_code,
            "decoding_type": decoding
        }
        print("✅ Transcription complete!")
        return result

    except Exception as e:
        print(f"⚠️ Transcription failed: {e}")
        return {"error": str(e), "transcript": "", "detected_language": lang_code}

# ----------------------------
# CLI Test
# ----------------------------

if __name__ == "__main__":
    test_audio = "./utils/user_query.wav" # replace with your local test file
    #print(os.path)
    if not os.path.exists(test_audio):
        print("⚠️ Please place a sample audio file named 'sample_hi.wav' in the folder.")
    else:
        out = transcribe_ai4bharat(test_audio, lang_code="hi", decoding="rnnt")
        print(out)
