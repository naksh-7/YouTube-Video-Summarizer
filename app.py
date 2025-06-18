"""
1. Try YouTube Data API captions (needs GOOGLE_API_KEY).
2. If no captions or quota error, download audio and
   run Whisper-small locally for transcription.
3. Summarise with BART-CNN as before.
"""

import os, re, json, subprocess, tempfile, requests
import streamlit as st
from transformers import AutoTokenizer, pipeline
from googleapiclient.discovery import build           # pip install google-api-python-client
from pytube import YouTube                            # pip install pytube
import whisper                                        # pip install -U openai-whisper

# ---------- constants ----------
MODEL      = "sshleifer/distilbart-cnn-12-6"
YT_API_KEY = os.getenv("GOOGLE_API_KEY")              # put in render/fly secrets
MAX_TOKENS = 950

tokenizer  = AutoTokenizer.from_pretrained(MODEL)
summariser = pipeline("summarization", model=MODEL, device=-1)

# ---------- helpers ----------
def extract_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def fetch_captions(video_id: str) -> str | None:
    if not YT_API_KEY:
        return None
    yt = build("youtube", "v3", developerKey=YT_API_KEY)
    tracks = yt.captions().list(part="id", videoId=video_id).execute()
    if not tracks.get("items"):
        return None
    track_id = tracks["items"][0]["id"]
    sub = yt.captions().download(id=track_id, tfmt="ttml").execute()
    # strip XML tags
    text = re.sub(r"<[^>]+>", " ", sub.decode("utf-8"))
    return " ".join(text.split())

def whisper_transcribe(url: str) -> str:
    yt = YouTube(url)
    audio = yt.streams.get_audio_only().download(filename=tempfile.mktemp(suffix=".mp4"))
    result = whisper.load_model("small").transcribe(audio, fp16=False)
    return result["text"]

def get_transcript(url: str) -> str:
    vid = extract_video_id(url)
    if not vid:
        raise ValueError("Invalid YouTube URL.")
    text = fetch_captions(vid)
    if text:
        return text
    # fallback to Whisper
    return whisper_transcribe(url)

def split_by_tokens(text, max_tokens=MAX_TOKENS):
    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(ids), max_tokens):
        yield tokenizer.decode(ids[i : i + max_tokens])

def summarise(text: str) -> str:
    parts = [summariser(chunk, max_length=180, min_length=30, do_sample=False)[0]["summary_text"]
             for chunk in split_by_tokens(text)]
    return " ".join(parts)

# ---------- UI ----------
st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")
st.title("🎥 YouTube Video Summarizer")

url = st.text_input("Enter a YouTube video URL")
if st.button("Summarize") and url:
    with st.spinner("Fetching transcript …"):
        try:
            transcript = get_transcript(url)
            with st.spinner("Summarising …"):
                st.text_area("Summary", summarise(transcript), height=300)
        except Exception as e:
            st.error(f"Failed: {e}")
