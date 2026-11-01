# AI Video Agent

An AI-powered Video Audio Processing and RAG Agent built with LangChain, Google Gemini, Whisper, and yt-dlp.

## Features
- **Audio Extraction**: Download YouTube audio or process local audio/video files.
- **Audio Processing**: Convert audio to 16kHz mono WAV and chunk for processing.
- **Transcription**: Local Whisper transcription with multi-language translation support.
- **Summarization & Title Generation**: Powered by Google Gemini (`gemini-1.5-flash` / `gemini-2.0-flash`).
- **Information Extraction**: Extract actionable items, key decisions, and questions using Google Gemini.

## Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   cd ai-video-agent
   ```
2. Configure environment variables in `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   WHISPER_MODEL=small
   GEMINI_MODEL=gemini-1.5-flash
   ```
3. Install dependencies:
   ```bash
   uv sync
   # or with pip:
   pip install -r Requirements.txt
   ```
