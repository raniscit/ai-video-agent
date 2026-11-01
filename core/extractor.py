#actionable items, decision, questions

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)


def build_chain(system_prompt: str, human_prompt: str):
    llm = get_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    return prompt_template | llm | StrOutputParser()


def extract_actionable_items(transcript: str) -> str:
    system_prompt = "You are an expert in extracting actionable items from text."
    human_prompt = "Extract actionable items from the following transcript:\n{input_text}"
    chain = build_chain(system_prompt, human_prompt)
    return chain.invoke({"input_text": transcript})

def extract_decisions(transcript: str) -> str:
    system_prompt = "You are an expert in extracting decisions from text."
    human_prompt = "Extract decisions from the following transcript:\n{input_text}"
    chain = build_chain(system_prompt, human_prompt)
    return chain.invoke({"input_text": transcript})

def extract_questions(transcript: str) -> str:
    system_prompt = "You are an expert in extracting questions from text."
    human_prompt = "Extract questions from the following transcript:\n{input_text}"
    chain = build_chain(system_prompt, human_prompt)
    return chain.invoke({"input_text": transcript})