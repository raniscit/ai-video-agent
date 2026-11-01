from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str) -> dict:
    print("Starting AI Video Assistant")

    chunks = process_input(source)

    if not chunks:
        raise ValueError("No audio chunks generated.")

    transcript = transcribe_all(chunks)

    if not transcript.strip():
        raise ValueError("Transcription is empty.")

    print(f"Raw transcription: {transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_actionable_items(transcript)
    decisions = extract_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()

    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\nChat with your video (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")