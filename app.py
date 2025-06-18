from transformers import AutoTokenizer, pipeline
import re
from youtube_transcript_api import YouTubeTranscriptApi
# import torch
import streamlit as st
# import gradio as gr
MODEL = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
summarizer = pipeline("summarization", model=MODEL, device=-1)   # use fp32/16 automatically, -1 for CPU change it to 0 for GPU as streamlit doesnt provide GPU had to change it to CPU

MAX_TOKENS = 950

def split_by_tokens(text, max_tokens=MAX_TOKENS):
    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(ids), max_tokens):
        yield tokenizer.decode(ids[i:i+max_tokens])

def summarise_transcript(video_url):
    vid = re.search(r'(?:v=|/)([A-Za-z0-9_-]{11})', video_url).group(1)
    transcript = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])
    full_text = " ".join(seg["text"] for seg in transcript)

    parts = [summarizer(chunk, max_length=180, min_length=30,
                        do_sample=False)[0]["summary_text"]
             for chunk in split_by_tokens(full_text)]
    return " ".join(parts)

video_url = "https://youtu.be/l00VBUXl1Q4?list=PLhr0Ua8H1x-K7UMXXeSfjULEIBCE1FVd1"
print(summarise_transcript(video_url))


st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")
st.title("🎥 YouTube Video Summarizer")

video_url = st.text_input("Enter a YouTube video URL:")

if st.button("Summarize"):
    with st.spinner("Summarizing transcript..."):
        summary = summarise_transcript(video_url)
        st.text_area("Summary:", summary, height=300)
