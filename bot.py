# OS related imports
import os
from glob import glob
from dotenv import load_dotenv

import logging

# import tqdm
from typing import List
import datetime

# Logger
from logger import CustomFormatter

# Store and retriever imports
# from langchain_community.multi_query import MultiQueryRetriever  """Use if needed. Adds complexity overhead"""
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

# LLM imports
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
import nltk

nltk.download("punkt_tab")

# Get all environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Set up logger
logging.basicConfig(filename=f"log/log-{datetime.datetime.today()}", level=logging.INFO)
log = logging.getLogger()
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
log.addHandler(ch)

# Constants
chunk_size = 700
chunk_overlap = 100


def load_srt_function(file_path: str) -> str:
    """
    Function to convert .srt files into transcripts.
    Input: file_path: str path of where the file exists
    Output: transcript string extracted from the .srt file
    """

    log.info(f"Loading {file_path.split('/')[-1]}")
    transcript = ""
    if not os.path.exists(file_path):
        print(f"File does not exist. Please check file path: {file_path}")
    with open(file_path) as file:
        for line in file:
            # Check if the line is a timestamp. If yes, continue
            if line[0].isnumeric():
                continue

            # If the line is empty, continue
            if line.strip() == "":
                continue

            # If it is not a timestamp, add the contents of the line to the transcript string.
            transcript += line.strip() + " "

    return transcript

    print
    return " ".join(t for t in transcript)


def create_vectorstore(files: List[str]):
    log.info("Creating vectorstores...")
    print("Creating vectorstores")
    documents = [load_srt_function(file_path) for file_path in files]
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs = []
    for document in documents:
        # Create langchain document for each document
        doc = Document(page_content=document)
        split_docs = text_splitter.split_documents([doc])
        docs.extend(split_docs)

    # embedding_function = HuggingFaceBgeEmbeddings(
    #     model_name="all-MiniLM-L6-v2",
    #     # model_kwargs={"device": "cpu"}, # Not needed since it does this internally. Can specify cpu/gpu for developer clarity.
    #     encode_kwargs={"normalize_embeddings": True},
    # )

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    """    vector store    """

    # TODO: Need to figure a way to use the persist persist_directory and not re-initialize it everytime the code is run.
    # Once we have all data embedded in the db, we can call the db directly into the retriever, which should reduce the startup time.
    db = Chroma.from_documents(
        docs, embedding_function, persist_directory="google-embed/transcripts.db"
    )

    retriever = db.as_retriever()

    log.info("Finished creating vectorstores...")
    print("Done creating vectorstores")
    return retriever


def load_vectorstore(persist_directory: str = None):
    if not persist_directory:
        print("No persistant storage found.")
        return

    # embedding_function = HuggingFaceBgeEmbeddings(
    #     model_name="all-MiniLM-L6-v2",
    #     # model_kwargs={"device": "cpu"}, # Not needed since it does this internally. Can specify cpu/gpu for developer clarity.
    #     encode_kwargs={"normalize_embeddings": True},
    # )

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_function
    )

    retriever = db.as_retriever()

    return retriever


def create_llm(
    model_name: str = "gemini-2.0-flash-exp",
    temperature: float = 1,
    top_p: float = 0.95,
):
    llm = GoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        task_type="CONVERSATIONAL",
    )
    return llm


def get_standalone_template():
    standalone_question_template = """
    You are an expert at understanding and rephrasing questions. Given a conversation transcript from course lectures and a follow-up question, rewrite the follow-up question as a standalone question that can be understood without prior context. Ensure the standalone question is clear, concise, and suitable for a 12th-grade reading level. 

Use the following format to structure your response:

Chat History:
{chat_history}

Follow-Up Input:
{question}

Standalone Question (Rephrased): 
- [Provide the rephrased question]

Tips for writing the standalone question:
1. Include any missing context from the chat history needed to make the question meaningful on its own.
2. Keep the language simple but formal, avoiding unnecessary jargon unless explained within the question.
3. Ensure the question stays focused and concise without losing critical details.

Chat History:
{chat_history}

Follow-Up Input:
{question}

Standalone Question (Rephrased):
- [Your rephrased question here]
"""

    standalone_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=standalone_question_template,
    )

    return standalone_question_prompt


def create_conversation_chain(retriever) -> ConversationalRetrievalChain:
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     output_key="answer",
    #     input_key="question",
    # )

    memory = ConversationSummaryBufferMemory(
        llm=create_llm(temperature=0.1),
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
        max_token_limit=1000,
    )

    standalone_question_prompt = get_standalone_template()

    # ? Prompt passed to the llm based on which it generates the answer. Modify this to align outputs as per requirement.
    general_system_template = r"""Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use five sentences maximum. Always provide source from document in the answer. Ensure you include examples that make it easy to understand the concept even for a complete newbie.
    {context}
    Question: {question}
    Helpful Answer:"""

    messages = [
        ("system", general_system_template),
        ("user", "{question}"),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=standalone_question_prompt,
        condense_question_llm=create_llm(temperature=0.3),
        memory=memory,
        retriever=retriever,
        llm=create_llm(temperature=0.6),
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        chain_type="stuff",
        verbose=False,
        return_source_documents=True,
        response_if_no_docs_found="I don't have enough information.",
    )

    return chain


def call_chain(user_input: str, chain: ConversationalRetrievalChain) -> str:
    log.info(f"Calling chain with {user_input=}")
    result = chain.invoke({"question": user_input})
    # print(result)
    logging.debug(result)
    return result


def main(query):
    persist_directory = "google-embed/transcripts.db"
    files = [f for f in glob("./data/*.srt")]
    if os.path.exists(persist_directory):
        retriever = load_vectorstore(persist_directory)
        log.info("Found existing vectorstore... Loading into it...")
    else:
        retriever = create_vectorstore(files)
    chain = create_conversation_chain(retriever)
    log.info("Chain built... Happy querying!")

    response = call_chain(query, chain)
    log.info(f"Returning response: {response}")
    return response


def test_run():
    query = "When starting your own content marketing, what is important?"
    persist_directory = "google-embed/transcripts.db"
    files = [f for f in glob("./data/*.srt")]
    if os.path.exists(persist_directory):
        retriever = load_vectorstore(persist_directory)
        log.info("Found existing vectorstore... Loading into it...")
    else:
        retriever = create_vectorstore(files)
    chain = create_conversation_chain(retriever)
    log.info("Chain built... Happy querying!")

    # Test questions
    # query = "What is TOFU content and how to identify it?"
    # response = call_chain(query, chain)

    # print(f"{query=}")
    # print(f"{response=}")

    while True:
        query = input("Query: ")
        if query in ["q", "exit"]:
            break
        response = call_chain(query, chain)
        result = response["answer"]
        print(f"{query=}")
        print(f"{result=}")


# test_run()


"""

"""
