"""
rag_pipeline.py
Core Retrieval-Augmented Generation pipeline for chatting with a YouTube video.

Pipeline stages:
    1. Indexing   - fetch transcript, split into chunks, embed, store in FAISS
    2. Retrieval  - pull the most relevant chunks for a given question
    3. Augmentation - inject retrieved chunks into a prompt template
    4. Generation - ask the LLM to answer using only that context
"""

from __future__ import annotations

from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .transcript_loader import fetch_transcript

DEFAULT_PROMPT = PromptTemplate(
    template="""You are a helpful assistant that answers questions about a YouTube video.
Answer ONLY using the provided transcript context below.
If the context does not contain enough information to answer, say you don't know
instead of guessing.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"],
)


class YouTubeRAG:
    """
    Builds a retrieval-augmented Q&A / summarization pipeline over a single
    YouTube video's transcript.

    Example:
        >>> rag = YouTubeRAG()
        >>> rag.build_index("Gfr50f6ZBvo")
        >>> rag.ask("What is this video about?")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        retriever_k: int = 4,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)

        self.vector_store: Optional[FAISS] = None
        self.retriever = None
        self.chain = None

    def build_index(self, video_id: str, languages: Optional[List[str]] = None) -> int:
        """
        Fetch the transcript, split it into chunks, embed them, and build the
        FAISS vector index + retrieval chain.

        Args:
            video_id: The YouTube video ID (not the full URL).
            languages: Preferred caption languages, in priority order.

        Returns:
            The number of chunks indexed.
        """
        transcript = fetch_transcript(video_id, languages=languages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.create_documents([transcript])

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.retriever_k}
        )
        self._build_chain()
        return len(chunks)

    def _build_chain(self) -> None:
        """Wire up the LangChain runnable: retrieve -> augment -> generate."""

        def format_docs(retrieved_docs) -> str:
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel(
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        self.chain = parallel_chain | DEFAULT_PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str) -> str:
        """Ask a question about the indexed video and get a grounded answer."""
        if self.chain is None:
            raise RuntimeError("Call build_index() before asking questions.")
        return self.chain.invoke(question)

    def summarize(self) -> str:
        """Generate a concise summary of the indexed video."""
        return self.ask("Please summarize this video in a clear and concise way.")
