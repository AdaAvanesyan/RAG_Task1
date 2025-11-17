# llm.py

import os
from typing import List, Tuple, Iterable
from openai import OpenAI

# Make sure OPENAI_API_KEY is set in your environment before running the server.
# For example in terminal:
# export OPENAI_API_KEY="sk-...."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(question: str, retrieved_chunks: List[Tuple[int, str]]) -> str:
    context = "\n\n".join([f"[Page {page}] {text}" for page, text in retrieved_chunks])
    prompt = f"""
You are a helpful assistant. Use ONLY the information in the CONTEXT below to answer the question.
If the answer is not in the context, say so clearly.
Always include page numbers as citations like [p.X].

CONTEXT:
{context}

QUESTION:
{question}
"""
    return prompt.strip()


def generate_answer(question: str, retrieved_chunks: List[Tuple[int, str]]) -> str:
    """
    Non-streaming LLM call.
    Returns the full answer as one string.
    """
    prompt = build_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()


def stream_answer(question: str, retrieved_chunks: List[Tuple[int, str]]) -> Iterable[str]:
    """
    Streaming LLM call.
    Yields small text pieces (tokens or token groups) one by one.
    This is designed to be wrapped by FastAPI's StreamingResponse.
    """
    prompt = build_prompt(question, retrieved_chunks)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta



# from typing import List, Dict
# from openai import OpenAI


# client = OpenAI()  # Reads OPENAI_API_KEY from environment


# def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
#     """
#     Generate a grounded answer for a question using retrieved document chunks.

#     The function constructs a prompt that includes the retrieved chunks as context
#     and asks the OpenAI chat model to answer the question strictly based on that
#     context. It also instructs the model to include page references like [p.X].

#     Args:
#         question (str): The user's question in natural language.
#         retrieved_chunks (List[Dict]): List of retrieved chunks, each with at least
#             'page' (int) and 'text' (str) keys.

#     Returns:
#         str: The generated answer as returned by the OpenAI model.
#     """
#     context = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks])
#     prompt = f"""
# You are a helpful assistant. Use only the information below to answer the question.
# If the answer is not present in the context, say so clearly.
# Always include page numbers as citations like [p.X].

# Context:
# {context}

# Question:
# {question}
# """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=600,
#     )
#     return response.choices[0].message.content.strip()

# from typing import List, Dict, Iterator
# from openai import OpenAI

# client = OpenAI()  # Reads OPENAI_API_KEY from environment


# def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
#     context = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks])
#     prompt = f"""
# You are a helpful assistant. Use only the information below to answer the question.
# If the answer is not present in the context, say so clearly.
# Always include page numbers as citations like [p.X].

# Context:
# {context}

# Question:
# {question}
# """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=600,
#     )
#     return response.choices[0].message.content.strip()


# def stream_answer(question: str, retrieved_chunks: List[Dict]) -> Iterator[str]:
#     """
#     Same as generate_answer, but yields the answer in small chunks suitable for streaming.
#     Each yield corresponds to a content delta from the OpenAI streaming API.
#     """
#     context = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks])
#     prompt = f"""
# You are a helpful assistant. Use only the information below to answer the question.
# If the answer is not present in the context, say so clearly.
# Always include page numbers as citations like [p.X].

# Context:
# {context}

# Question:
# {question}
# """
#     stream = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=600,
#         stream=True,   # this is the key part
#     )

#     for chunk in stream:
#         delta = chunk.choices[0].delta.content
#         if delta:
#             # Yield each small piece of text as it arrives
#             yield delta