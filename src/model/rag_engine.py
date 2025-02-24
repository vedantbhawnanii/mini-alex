# ---------------------------------------
# Imports
# ---------------------------------------

import logging
import os
import warnings
from glob import glob
from typing import List

import nltk
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import NLTKTextSplitter

from src.utils.logger import ColorLogger

nltk.download("punkt_tab")
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
if not os.path.isdir("log"):
    os.mkdir("log")

log = ColorLogger("my_app", "my_app.log")

# ---------------------------------------
# Data functions
# ---------------------------------------


class RagAgent:
    def __init__(self):
        self.CHUNK_SIZE = 1200
        self.CHUNK_OVERLAP = 200
        self.COMPRESSION_CHUNK_SIZE = 500
        self.COMPRESSION_CHUNK_OVERLAP = 20
        self.PERSIST_DIR = "./google-embed/transcripts.db/"

    def _load_srt_function(self, file_path: str) -> str:
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

    # ---------------------------------------
    #          Vectorstore Functions
    # ---------------------------------------

    def _create_vectorstore(self, files: List[str]):
        """
        Create a vector store.

        Arguements:
        files: str = List of files to be added to the vectorstore.
        api_key: str = API key for Google AI

        Returns: vectorstore, embedding function.
        """
        text_splitter = NLTKTextSplitter(
            chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP
        )
        docs = []
        for file in files:
            document = self._load_srt_function(file)
            doc = Document(page_content=document, metadata={"filename": file})
            split_docs = text_splitter.split_documents([doc])
            docs.extend(split_docs)
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        store = Chroma.from_documents(
            docs, embedding_function, persist_directory=self.PERSIST_DIR
        )
        log.info("Finished creating vectorstore...")
        return store, embedding_function

    def _load_vectorstore(self):
        """
        Loads a vectorstore from local dir.
        """
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        store = Chroma(
            persist_directory=self.PERSIST_DIR, embedding_function=embedding_function
        )
        return store, embedding_function

    # ---------------------------------------
    #         Retriever Functions
    # ---------------------------------------

    def _vectorstore_backed_retriever(
        self, vectorstore, search_type="similarity", k=16, score_threshold=None
    ):
        """create a vectorsore-backed retriever
        Parameters:
            search_type: Defines the type of search that the Retriever should perform.
                Can be "similarity" (default), "mmr", or "similarity_score_threshold"
            k: number of documents to return
            score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
        """
        search_kwargs = {}
        if k is not None:
            search_kwargs["k"] = k
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        retriever = vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        return retriever

    def _create_compression_retriever(
        self,
        embeddings,
        base_retriever,
        k=16,
        similarity_threshold=None,
    ) -> ContextualCompressionRetriever:
        """
        Parameters:
            embeddings: GoogleGenerativeAIEmbeddings
            base_retriever: a vectorstore-backed retriever.
            CHUNK_SIZE (int): Documents will be splitted into smaller chunks using a CharacterTextSplitter with a default CHUNK_SIZE of 500.
            k (int): top k relevant chunks to the query are filtered using the EmbeddingsFilter. default =16.
            similarity_threshold : minimum relevance threshold used by the EmbeddingsFilter.. default =None.
        """

        splitter = CharacterTextSplitter(
            chunk_size=self.COMPRESSION_CHUNK_SIZE,
            chunk_overlap=self.COMPRESSION_CHUNK_OVERLAP,
            separator=". ",
        )
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(
            embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
        )

        reordering = LongContextReorder()
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter, reordering]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )

        return compression_retriever

    def _get_retriever(
        self,
        store,
        embeddings,
        search_type="similarity",
        k=16,
        similarity_threshold=None,
    ):
        base_retriever = self._vectorstore_backed_retriever(
            store, search_type=search_type, k=k
        )
        compressed_retriever = self._create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            similarity_threshold=similarity_threshold,
            k=k,
        )
        return compressed_retriever

    # ---------------------------------------
    #              Prompting
    # ---------------------------------------

    def _get_standalone_template(self) -> PromptTemplate:
        standalone_question_template = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    Use the following format to structure your response:

    Chat History: {chat_history}

    Follow-Up Input: {question}

    Standalone Question: {{Your answer here}}
    """

        standalone_question_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=standalone_question_template,
        )

        return standalone_question_prompt

    def create_conversation_chain(self, retriever) -> ConversationalRetrievalChain:
        memory = ConversationSummaryBufferMemory(
            llm=self._create_llm(temperature=0.1),
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question",
            max_token_limit=2000,  ### INFO: Change this to increase answer size. In paid api calls, increasing this will increase cost per answer.
        )

        standalone_question_prompt = self._get_standalone_template()

        # INFO: Prompt passed to the llm based on which it generates the answer. Modify this to align outputs as per requirement.
        general_system_template = r"""You are an assistant for ImagineGrowth. You are capable of answering questions about their courses. All neccessary information is provided to you as data.
        You are also capable of answering to greetings by providing an introduction of yourself.
        Try to answer questions based on the information provided in the data content 
        below. If you are asked a question that isn't covered in the provided data, respond based on the given information 
        and your best judgment. If you don't know the answer reply that you don't know. 
        The answer formatting should be whatsapp friendly. If the answer contains a lot of information, provide it in a presentable format.

        Data: {context}
        Question: {question}
        Helpful Answer:"""

        messages = [
            ("system", general_system_template),
            ("user", "{question}"),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        chain = ConversationalRetrievalChain.from_llm(
            condense_question_prompt=standalone_question_prompt,
            condense_question_llm=self._create_llm(
                temperature=0.2,
            ),
            memory=memory,
            retriever=retriever,
            llm=self._create_llm(
                temperature=0.6,
            ),
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            chain_type="stuff",
            verbose=False,
            return_source_documents=True,
            response_if_no_docs_found="I cannot answer this question right now. Please ask me something else.",
        )

        return chain

    # ---------------------------------------
    #       Creating LLMs and chains.
    # ---------------------------------------

    def _create_llm(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.5,
        top_p: float = 0.95,
    ) -> GoogleGenerativeAI:
        llm = GoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
        )
        return llm

    def call_chain(self, user_input: str, chain: ConversationalRetrievalChain) -> str:
        log.info(f"Calling chain with {user_input=}")
        result = chain.invoke({"question": user_input})
        logging.info(result)

        # Extract filename from source documents (if any)
        # filenames = set()
        # if result.get("source_documents"):
        #     for doc in result["source_documents"]:
        #         filenames.add(doc.metadata.get("filename", "Unknown Source"))
        #
        # answer = result["answer"].strip()
        # if filenames:
        #     answer += f"\n\nSource: {', '.join(filenames)}"

        return result["answer"].strip()

    def get_chain(self) -> ConversationalRetrievalChain:
        if os.path.exists(self.PERSIST_DIR):
            log.info("Found existing vectorstore... Loading into it...")
            store, embedding_function = self._load_vectorstore()
            compressed_retriever = self._get_retriever(
                store=store,
                search_type="similarity",
                k=16,
                embeddings=embedding_function,
            )

        else:
            files = [f for f in glob("./data/transcripts/*.txt")]
            log.info("Creating new store")
            store, embedding_function = self._create_vectorstore(files)
            compressed_retriever = self._get_retriever(
                store=store,
                search_type="similarity",
                k=16,
                embeddings=embedding_function,
            )
        chain = self.create_conversation_chain(compressed_retriever)
        log.info("Chain built... Happy querying!")

        return chain


def main():
    rag = RagAgent()
    chain = rag.get_chain()
    while True:
        query = input("Query: ")
        response = rag.call_chain(query, chain=chain)
        print(f"Response:\n{response}")


# main()
