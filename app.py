import re
import requests
import yt_dlp
import streamlit as st
from transformers import AutoTokenizer, pipeline

# Model setup
MODEL = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
summarizer = pipeline("summarization", model=MODEL, device=-1)  # CPU mode

MAX_TOKENS = 950

def split_by_tokens(text, max_tokens=MAX_TOKENS):
    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(ids), max_tokens):
        yield tokenizer.decode(ids[i:i+max_tokens])

def summarise_transcript(video_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'writeautomaticsub': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if not subtitles or "en" not in subtitles:
                return "⚠️ No English subtitles found for this video."

            subtitle_url = subtitles["en"][0]["url"]
            vtt_text = requests.get(subtitle_url).text

            # Convert VTT to plain text
            lines = []
            for line in vtt_text.splitlines():
                if not line or "-->" in line or re.match(r"^\d+$", line):
                    continue
                lines.append(line.strip())

            full_text = " ".join(lines)

            parts = [summarizer(chunk, max_length=180, min_length=30, do_sample=False)[0]["summary_text"]
                     for chunk in split_by_tokens(full_text)]
            return " ".join(parts)

    except Exception as e:
        return f"🚫 Error extracting transcript: {str(e)}"


# Streamlit UI
st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")
st.title("🎥 YouTube Video Summarizer")
st.markdown("Enter a YouTube video URL to generate a concise summary of its transcript.")

video_url = st.text_input("📺 YouTube Video URL:")

if st.button("Summarize"):
    if not video_url:
        st.warning("Please enter a YouTube video URL.")
    else:
        with st.spinner("⏳ Summarizing transcript..."):
            summary = summarise_transcript(video_url)
            st.text_area("📝 Summary", summary, height=300)
