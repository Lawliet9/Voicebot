"""
translation.py
---------------
Handles:
1️⃣ Translation of multilingual user queries to English.
2️⃣ Topic extraction (maps user intent to known scheme categories).

Uses Google Gemini 2.5 Flash via the `google-generativeai` library.
"""

import os
import google.generativeai as genai

# ---------------------------------------
# Gemini Model Setup
# ---------------------------------------

# Configure Gemini API key (set in environment or directly here)
#genai.configure(api_key="AIzaSyCoI_9gL_iLoNcJMUz6g9Z-jxFVbMG98Bg")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------
# Translation Function
# ---------------------------------------

def translate_to_english(text: str) -> str:
    """
    Translate vernacular (Hindi, Marathi, Tamil, etc.) text to English using Gemini 2.5 Flash.
    Returns the English translation or the original text if translation fails.
    """
    if not text or not text.strip():
        return ""

    prompt = f"""
    Translate the following text to English accurately.
    Preserve the meaning, names, and tone.
    Only return the translated English sentence, nothing else.

    Text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        translated = response.text.strip()
        return translated
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return text  # fallback to original text


# ---------------------------------------
# Topic Extraction Function
# ---------------------------------------

def extract_topic_from_query(query: str, candidate_topics: list[str] = None) -> str:
    """
    Identify the main topic or category of the user query using Gemini.
    If candidate topics are provided, it classifies among them.

    Args:
        query: user query (in English)
        candidate_topics: list of categories from dataset (optional)
    Returns:
        str: best-matching topic name
    """
    if not query or not query.strip():
        return ""

    if candidate_topics:
        topics_str = ", ".join(candidate_topics)
        prompt = f"""
        The user asked: "{query}"

        Choose the most relevant topic from this list:
        [{topics_str}]

        Only return one topic name from the list.
        """
    else:
        prompt = f"""
        Identify the main topic or category of the following query in one or two words.
        Return only the topic.

        Query:
        {query}
        """

    try:
        response = model.generate_content(prompt)
        topic = response.text.strip().split("\n")[0]
        return topic
    except Exception as e:
        print(f"⚠️ Topic extraction failed: {e}")
        return ""


# ---------------------------------------
# CLI Test
# ---------------------------------------

if __name__ == "__main__":
    sample_text = "किसान क्रेडिट कार्ड योजना के बारे में बताओ"
    print("🗣️ Original:", sample_text)

    english = translate_to_english(sample_text)
    print("🌐 English:", english)

    topics = ["agri_credit", "financial_inclusion", "housing", "education"]
    topic = extract_topic_from_query(english, topics)
    print("🧭 Detected Topic:", topic)
