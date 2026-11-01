
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_decisions, extract_questions

load_dotenv()


source = "https://www.youtube.com/watch?v=6uUblznfrsk"

print("Processing video...")
chunks = process_input(source)

print(f"\nTotal chunks: {len(chunks)}")

print("\nTranscribing audio using Whisper...")
transcript = transcribe_all(chunks)

print("\n=== TRANSCRIPT ===\n")
print(transcript)


print("\n=== SUMMARY ===\n")
summary = summarize(transcript)
print(summary)

print("\n=== TITLE ===\n")
title = generate_title(transcript)
print(title)

print("\n=== ACTIONABLE ITEMS ===\n")
actionable_items = extract_actionable_items(transcript)
print(actionable_items)

print("\n=== DECISIONS ===\n")
decisions = extract_decisions(transcript)
print(decisions)

print("\n=== QUESTIONS ===\n")
questions = extract_questions(transcript)
print(questions)
