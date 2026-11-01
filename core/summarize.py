from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os


def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.3)

def split_transcript(transcript: str, chunk_size: int = 5000, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(transcript)


def summarize(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text in a concise manner, highlighting key points and important details."),
        ("human", "Summarize the following text:\n{input_text}"),
    ])
 
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []
    for chunk in chunks:
        summary = map_chain.invoke({"input_text": chunk})
        chunk_summaries.append(summary)

    combined_summary = "\n\n ".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer. Combine the following summaries into a single, coherent summary, ensuring clarity and conciseness."),
    ("human", "Combine the following summaries:\n{input_text}"),
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    finally_summary = combined_chain.invoke({"input_text": combined_summary})

    return finally_summary


def generate_title(transcript : str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in generating concise and relevant titles for given text."),
        ("human", "Generate a concise and relevant title for the following text:\n{input_text}"),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    title = title_chain.invoke({"input_text": transcript})

    return title