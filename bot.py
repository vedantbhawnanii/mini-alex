# OS related imports
import os
from glob import glob
from chromadb import api
from dotenv import load_dotenv
import warnings


from typing import List, Dict
import datetime

# Logger
from logger import CustomFormatter
import logging

from langchain_text_splitters import NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory

# LLM imports
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

import nltk

### Document Compressor Pipeline
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

nltk.download("punkt_tab")
warnings.filterwarnings("ignore", category=FutureWarning)

# Get all environment variables
# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Set up logger
if not os.path.isdir("log"):
    os.mkdir("log")
logging.basicConfig(filename=f"log/log-{datetime.datetime.today()}", level=logging.INFO)
log = logging.getLogger()
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
log.addHandler(ch)

# Constants
chunk_size = 1200
chunk_overlap = 200

# INFO:                  Data functions                     

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

# INFO:                   Vectorstore Functions

def create_vectorstore(files: List[str], api_key):
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

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = api_key)

    """    vector store    """

    store = Chroma.from_documents(
        docs, embedding_function, persist_directory="google-embed/transcripts.db"
    )


    log.info("Finished creating vectorstore...")
    return store, embedding_function


def load_vectorstore(persist_directory: str, api_key):
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    store = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_function
    ) 

    return store, embedding_function 

# INFO:                     Retriever Functions

def vectorstore_backed_retriever(vectorstore,search_type="similarity",k=4,score_threshold=None):
    """create a vectorsore-backed retriever
    Parameters: 
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4) 
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever

def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a vectorstore-backed retriever) into a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    into smaller chunks, removes redundant documents, filters out the most relevant documents,
    and reorder the documents so that the most relevant are at the top and bottom of the list.
    
    Parameters:
        embeddings: GoogleGenerativeAIEmbeddings 
        base_retriever: a vectorstore-backed retriever.
        chunk_size (int): Documents will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500. 
        k (int): top k relevant chunks to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : minimum relevance threshold used by the EmbeddingsFilter.. default =None.
    """
    
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")    
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold) # similarity_threshold and top K

    # Less relevant document will be at the middle of the list and more relevant elements at the beginning or end of the list.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]  
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, 
        base_retriever=base_retriever
    )

    return compression_retriever

# INFO:                   Prompting                 

def get_standalone_template():
    standalone_question_template = """You are an expert at understanding and rephrasing questions. Given a conversation transcript from course lectures and a follow-up question, rewrite the follow-up question as a standalone question that can be understood without prior context. Ensure the standalone question is clear, concise, and provides all necessary context.

Use the following format to structure your response:

Chat History: {chat_history}

Follow-Up Input: {question}

Standalone Question:  
"""

    standalone_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=standalone_question_template,
    )

    return standalone_question_prompt

def create_conversation_chain(retriever, api_key) -> ConversationalRetrievalChain:
    memory = ConversationSummaryBufferMemory(
        llm=create_llm(temperature=0.1, api_key=api_key),
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
        max_token_limit=1000,
    )

    standalone_question_prompt = get_standalone_template()

    # ? Prompt passed to the llm based on which it generates the answer. Modify this to align outputs as per requirement.
    general_system_template = r"""Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Always provide source from document in the answer. Ensure you include examples that make it easy to understand the concept even for a complete newbie.
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
        condense_question_llm=create_llm(temperature=0.3, api_key=api_key),
        memory=memory,
        retriever=retriever,
        llm=create_llm(temperature=0.6, api_key=api_key),
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        chain_type="stuff",
        verbose=False,
        return_source_documents=True,
        response_if_no_docs_found="I cannot answer this question right now. Please ask me something else.",
    )

    return chain

# INFO:                     Creating LLMs and chains.

def create_llm(
    api_key,
    model_name: str = "gemini-2.0-flash-exp",
    temperature: float = 0.5,
    top_p: float = 0.95,
):
    llm = GoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        api_key = api_key
    )

    return llm


def call_chain(user_input: str, chain: ConversationalRetrievalChain) -> str:
    log.info(f"Calling chain with {user_input=}")
    result = chain.invoke({"question": user_input})
    # print(result)
    logging.debug(result)
    return result

# INFO:                        Streamlit

def main(api_key):
    persist_directory = "google-embed/transcripts.db"

    if os.path.exists(persist_directory):
        log.info("Found existing vectorstore... Loading into it...")
        store, embedding_function = load_vectorstore(persist_directory=persist_directory, api_key=api_key)
        base_retriever = vectorstore_backed_retriever(store, "similarity", k=10) 
        compressed_retriever = create_compression_retriever(
            embeddings = embedding_function, 
            base_retriever = base_retriever,
        ) 
          
    else:
            
        files = [f for f in glob("./data/*.srt")]
        store, embedding_function = create_vectorstore(files, api_key= api_key)
        base_retriever = vectorstore_backed_retriever(store, "similarity", k=10) 
        compressed_retriever = create_compression_retriever(
            embeddings = embedding_function, 
            base_retriever = base_retriever,
        )

    chain = create_conversation_chain(compressed_retriever, api_key=api_key)
    log.info("Chain built... Happy querying!")

    return chain

# INFO:                     Test Function.

def test_run(api_key):
    query = "When starting your own content marketing, what is important?"
    persist_directory = "google-embed/transcripts.db"
    files = [f for f in glob("./data/*.srt")]

    
    if os.path.exists(persist_directory):
        log.info("Found existing vectorstore... Loading into it...")
        store, embedding_function = load_vectorstore(persist_directory=persist_directory, api_key=api_key)
        base_retriever = vectorstore_backed_retriever(store)
        compressed_retriever = create_compression_retriever(
            embeddings = embedding_function, 
            base_retriever = base_retriever,
        ) 

    else:
        
        store, embedding_function = create_vectorstore(files, api_key=api_key)
        base_retriever = vectorstore_backed_retriever(store)
        compressed_retriever = create_compression_retriever(
            embeddings = embedding_function, 
            base_retriever = base_retriever,
        ) 
    
    chain = create_conversation_chain(compressed_retriever, api_key=api_key)

    # Test questions
    query = "What is TOFU, MOFU, BOFU content and how to identify it?"
    response = call_chain(query, chain)

    print(f"{query=}")
    print(f"Answer: {response["answer"]}")

    query = "How many states are there in United States?"
    response = call_chain(query, chain)

    print(f"{query=}")
    print(f"Answer: {response["answer"]}")


    # while True:
    #     query = input("Query: ")
    #     if query in ["q", "exit"]:
    #         break
    #     response = call_chain(query, chain)
    #     result = response["answer"]
    #     print(f"{query=}")
    #     print(f"{result=}")

test_run("AIzaSyCvsLIQzh6bAXQlx_f7KCaxDWKGRSKMEyo")
