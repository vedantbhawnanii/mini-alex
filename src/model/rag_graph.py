# ---------------------------------------
# Imports
# ---------------------------------------
import os
import uuid
from glob import glob
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph

# ---------------------------------------
# State and Graph Definition
# ---------------------------------------

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


class GraphState(TypedDict):
    messages: List[BaseMessage]
    question: str
    documents: List[Document]
    filenames: set
    response: Optional[str] = None


class LangGraphRAG:
    def __init__(self):
        self.CHUNK_SIZE = 1200
        self.CHUNK_OVERLAP = 200
        self.COMPRESSION_CHUNK_SIZE = 500
        self.COMPRESSION_CHUNK_OVERLAP = 20
        self.PERSIST_DIR = "../../data/google-embed/transcripts.db/"

        # Initialize core components
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = self._init_vectorstore()
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.6)
        self.retriever = self._init_retriever()

        # Build LangGraph workflow
        self.workflow = self._build_graph()

    # ---------------------------------------
    # Vectorstore Initialization (From Original)
    # ---------------------------------------
    def _load_srt(self, file_path: str) -> str:
        transcript = ""
        with open(file_path) as file:
            for line in file:
                if line[0].isnumeric() or line.strip() == "":
                    continue
                transcript += line.strip() + " "
        return transcript

    def _create_vectorstore(self, files: List[str]) -> Chroma:
        docs = []
        for file in files:
            content = self._load_srt(file)
            doc = Document(page_content=content, metadata={"filename": file})
            docs.append(doc)
        return Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.PERSIST_DIR,
        )

    def _init_vectorstore(self) -> Chroma:
        if os.path.exists(self.PERSIST_DIR):
            return Chroma(
                persist_directory=self.PERSIST_DIR, embedding_function=self.embeddings
            )
        return self._create_vectorstore(glob("./data/*.txt"))

    # ---------------------------------------
    # Retriever Setup (From Original)
    # ---------------------------------------
    def _create_compression_retriever(self) -> ContextualCompressionRetriever:
        splitter = CharacterTextSplitter(
            chunk_size=self.COMPRESSION_CHUNK_SIZE,
            chunk_overlap=self.COMPRESSION_CHUNK_OVERLAP,
            separator=". ",
        )
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, k=16)
        reordering = LongContextReorder()

        return ContextualCompressionRetriever(
            base_compressor=DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter, reordering]
            ),
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 16}),
        )

    def _init_retriever(self) -> ContextualCompressionRetriever:
        return self._create_compression_retriever()

    # ---------------------------------------
    # LangGraph Construction
    # ---------------------------------------
    def _build_graph(self):
        builder = StateGraph(GraphState)

        # Define Nodes
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)
        builder.add_node("update_memory", self.update_memory)
        builder.add_node("format_response", self.format_response)

        # Define Edges
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "format_response")
        builder.add_edge("format_response", "update_memory")
        builder.add_conditional_edges(
            "update_memory", self.decide_continue, {"continue": "retrieve", "end": END}
        )

        return builder.compile()

    # ---------------------------------------
    # Node Implementations
    # ---------------------------------------
    def retrieve(self, state: GraphState) -> Dict:
        """Retrieve documents for question"""
        docs = self.retriever.invoke(state["question"])
        filenames = {doc.metadata["filename"] for doc in docs}
        return {"documents": docs, "filenames": filenames}

    def generate(self, state: GraphState) -> Dict:
        """Generate response using LLM"""
        prompt = self._build_prompt(state["messages"], state["documents"])
        response = self.llm.invoke(prompt)
        return {"response": response}

    def format_response(self, state: GraphState) -> Dict:
        """Format response with sources"""
        response = state["response"]
        if state["filenames"]:
            response += f"\n\nSource: {', '.join(state['filenames'])}"
        return {"response": response.strip()}

    def update_memory(self, state: GraphState) -> Dict:
        """Update conversation history"""
        new_messages = state["messages"] + [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["response"]),
        ]
        return {"messages": new_messages}

    # ---------------------------------------
    # Helper Methods (From Original)
    # ---------------------------------------
    def _build_prompt(
        self, messages: List[BaseMessage], docs: List[Document]
    ) -> ChatPromptTemplate:
        system_template = """Answer questions using these documents:
        {docs}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        Helpful Answer:"""

        return ChatPromptTemplate.from_template(system_template).format(
            docs="\n".join(d.page_content for d in docs),
            chat_history="\n".join(m.content for m in messages),
            question=messages[-1].content if messages else "",
        )

    def decide_continue(self, state: GraphState) -> str:
        """Check if conversation should continue"""
        last_message = state["messages"][-1].content.lower()
        if any(keyword in last_message for keyword in ["exit", "goodbye", "stop"]):
            return "end"
        return "continue"

    # ---------------------------------------
    # Public Interface
    # ---------------------------------------
    def chat(
        self,
        question: str,
        thread_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Execute conversation thread"""
        thread_id = thread_id or str(uuid.uuid4())
        result = self.workflow.invoke(
            {"question": question, "messages": []},
            RunnableConfig(thread_id=thread_id, **(config or {})),
        )
        return result["response"]

    def continue_chat(
        self, question: str, thread_id: str, config: Optional[RunnableConfig] = None
    ) -> str:
        """Continue existing conversation thread"""
        result = self.workflow.invoke(
            {"question": question},
            RunnableConfig(thread_id=thread_id, **(config or {})),
        )
        return result["response"]


# ---------------------------------------
# Usage Example
# ---------------------------------------
if __name__ == "__main__":
    rag = LangGraphRAG()

    # User 1 conversation
    thread_id_1 = str(uuid.uuid4())
    print(rag.chat("What's in the documents?", thread_id_1))
    print(rag.continue_chat("Can you elaborate?", thread_id_1))

    # User 2 conversation
    thread_id_2 = str(uuid.uuid4())
    print(rag.chat("Different question?", thread_id_2))
