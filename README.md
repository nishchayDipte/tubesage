# TubeSage — Chat With Any YouTube Video Using RAG

TubeSage lets you **ask questions about a YouTube video's content** and get answers
grounded strictly in that video's transcript — no hallucinated facts, no guessing.
It uses a classic **Retrieval-Augmented Generation (RAG)** pipeline built with
LangChain, OpenAI, and FAISS.

> Paste a video ID → TubeSage fetches the transcript, indexes it, and lets you
> chat with it or get an instant summary.

---

##  Features

-  **Works on any YouTube video with captions** (auto-generated or manual)
-  **Semantic search** over the transcript using FAISS vector store
-  **Grounded Q&A** — answers only from the transcript, says "I don't know" otherwise
-  **One-command summarization**
-  **CLI with interactive chat mode**
-  **Clean, modular pipeline** you can reuse in your own apps (importable as a library)

---

##  How It Works (RAG Pipeline)

```
YouTube Video ID
       │
       ▼
┌─────────────────┐
│ 1. Transcript   │  youtube-transcript-api fetches captions
│    Loader       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Text Splitter│  RecursiveCharacterTextSplitter
│   (chunking)    │  chunk_size=1000, overlap=200
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. Embeddings + │  OpenAIEmbeddings → FAISS vector store
│    Vector Store │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. Retriever    │  similarity search, top-k relevant chunks
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Prompt +     │  context + question → PromptTemplate
│LLM (GPT-4o-mini)│ → grounded answer
└────────┬────────┘
         ▼
     Your Answer
```

This is the standard **Index → Retrieve → Augment → Generate** RAG pattern.

---

##  Project Structure

```
tubesage/
├── tubesage/
│   ├── __init__.py          # Package exports
│   ├── transcript_loader.py # Fetches & flattens YouTube transcripts
│   ├── rag_pipeline.py      # Core RAG pipeline (YouTubeRAG class)
│   └── cli.py               # Command-line interface
├── main.py                  # Entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/tubesage.git
cd tubesage
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example env file and add your own OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=sk-your-real-key-here
```


> Get a key at https://platform.openai.com/api-keys

---

## 🧑‍💻 Usage

### Interactive chat mode

```bash
python main.py Gfr50f6ZBvo
```

```
Fetching transcript and building index for video 'Gfr50f6ZBvo'...
Indexed 142 chunks. Ready!

Ask questions about the video below.
Type 'summary' for a summary, or 'exit' / 'quit' to stop.

You: Who is Demis Hassabis?
Bot: ...

You: summary
Bot: ...
```

### One-off question (non-interactive)

```bash
python main.py Gfr50f6ZBvo -q "Is nuclear fusion discussed in this video?"
```

### Use a different transcript language

```bash
python main.py <video_id> --lang hi en
```

### Use it as a Python library

```python
from tubesage import YouTubeRAG

rag = YouTubeRAG()
rag.build_index("Gfr50f6ZBvo")

print(rag.ask("What is DeepMind?"))
print(rag.summarize())
```

---

## 🛠️ Tech Stack

| Component        | Tool                                   |
|-------------------|-----------------------------------------|
| LLM               | OpenAI `gpt-4o-mini` (via LangChain)    |
| Embeddings        | OpenAI `text-embedding-3-small`         |
| Vector Store      | FAISS                                   |
| Orchestration     | LangChain (Runnables / LCEL)            |
| Transcript Source | `youtube-transcript-api`                |
| Config            | `python-dotenv`                         |

---

## 🔮 Future Improvements

- [ ] Support multiple videos / playlists in one knowledge base
- [ ] Add a Streamlit / Gradio web UI
- [ ] Swap FAISS for a persistent vector DB (Chroma / Pinecone)
- [ ] Add streaming responses
- [ ] Add automatic language detection and translation
- [ ] Add unit tests with mocked transcripts

---

##  Notes & Limitations

- Only works on videos that have captions/subtitles available.
- Answers are limited to what's actually said in the video — TubeSage will not
  use outside knowledge to answer.
- Requires a valid OpenAI API key with available credits.

---


## Author

Built as a learning project to demonstrate a practical RAG pipeline using
LangChain, OpenAI, and FAISS. Feedback and PRs welcome!
