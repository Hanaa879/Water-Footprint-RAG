# Water-Footprint-RAG
 PDF RAG with Groq & LangChain

A lightweight Retrieval-Augmented Generation (RAG) script that loads a PDF report, splits it into manageable chunks, and uses Groq's ultra-fast LLM inference (Llama 3.3 70B) via LangChain to answer questions grounded in the document's content.

This example uses Microsoft's 2024 Environmental Sustainability Report as the source document, but works with any PDF.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Code Explanation](#code-explanation)
6. [Output](#output)
7. [Notes & Limitations](#notes--limitations)

---

## Overview

This script demonstrates a minimal RAG pipeline in three steps:

1. **Load** — a PDF is parsed into per-page documents
2. **Chunk** — the pages are split into smaller, overlapping text chunks suitable for feeding to an LLM
3. **Query** — a single chunk is used as context for a question sent to Groq's `llama-3.3-70b-versatile` model, which answers based only on that retrieved context

It's designed to run in **Google Colab**, using Colab's Secrets feature (`userdata`) to securely store the Groq API key rather than hardcoding it.

---

## How It Works

```text
[ PDF File ] 
     │
     ▼
[ PyPDFLoader ]  →  loads PDF into per-page Document objects
     │
     ▼
[ RecursiveCharacterTextSplitter ]  →  splits pages into 1000-char chunks (100-char overlap)
     │
     ▼
[ Select a chunk ]  →  chunks[100].page_content used as context
     │
     ▼
[ ChatGroq (Llama 3.3 70B) ]  →  answers a question grounded in that chunk
     │
     ▼
[ Printed response ]
```

---

## Prerequisites

- A [Groq API key](https://console.groq.com/keys) (free tier available)
- Google Colab (this script relies on `google.colab.userdata` for secret management)
- A PDF file uploaded to your Colab session's file browser

---

## Setup

1. **Get a Groq API key** from [console.groq.com/keys](https://console.groq.com/keys).

2. **Add it to Colab Secrets:**
   - Click the 🔑 key icon in the left sidebar of Colab
   - Add a new secret named `GROQ_API_KEY`
   - Paste your key as the value, and toggle **Notebook access** on

3. **Upload your PDF** to the Colab session using the 📁 folder icon in the left sidebar. By default this script expects a file named `Microsoft-2024-Environmental-Sustainability-Report.pdf` in the root working directory — replace this with the path to your own PDF.

4. **Run the notebook cells in order.**

---

## Code Explanation

### Step 0: Installing Dependencies

```python
!pip install -q -U langchain-community langchain-groq pypdf langchain-text-splitters
```

Installs the four packages this script depends on, using Colab's shell-command syntax (`!`):
- **`langchain-community`** — provides `PyPDFLoader`, the document loader used to parse the PDF
- **`langchain-groq`** — provides `ChatGroq`, the LangChain wrapper around Groq's chat completion API
- **`pypdf`** — the underlying PDF-parsing library that `PyPDFLoader` uses internally
- **`langchain-text-splitters`** — provides `RecursiveCharacterTextSplitter`, used to break long documents into smaller chunks

The `-q` flag suppresses most install output (quiet mode), and `-U` upgrades these packages to their latest versions if already installed.

### Step 1: Imports

```python
import time
from google.colab import userdata
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
```

- **`time`** is imported but not actually used anywhere in this script — likely left over from an earlier version that included retry/delay logic, or reserved for future rate-limiting.
- **`userdata`** is Colab's secrets manager, used to securely retrieve the Groq API key without exposing it in the notebook's visible code or output.
- **`PyPDFLoader`**, **`RecursiveCharacterTextSplitter`**, and **`ChatGroq`** are the three core LangChain components driving the pipeline, as described in [How It Works](#how-it-works).

### Step 2: Initializing the Groq LLM

```python
raw_key = userdata.get('GROQ_API_KEY')
clean_key = raw_key.strip().split()[0] if raw_key else None

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=clean_key,
    temperature=0
)
```

- **`userdata.get('GROQ_API_KEY')`** fetches the secret stored in Colab's Secrets panel under that exact name.
- **`raw_key.strip().split()[0]`** is a defensive cleanup step: `.strip()` removes leading/trailing whitespace (including trailing newlines that can sometimes get pasted into the Secrets field), and `.split()[0]` takes only the first whitespace-separated token — guarding against a key accidentally containing a stray space or line break, which would otherwise cause an "Illegal header value" error when the key is sent as an HTTP Authorization header. The `if raw_key else None` guard prevents a crash if the secret wasn't found at all, falling back to `None` (which will still fail later, but with a clearer authentication error rather than an `AttributeError` on `None.strip()`).
- **`ChatGroq(...)`** instantiates the LangChain chat model wrapper, configured to:
  - Use **`llama-3.3-70b-versatile`**, a 70-billion-parameter Llama 3.3 model served on Groq's fast inference hardware
  - Authenticate with the cleaned API key
  - Use **`temperature=0`**, which makes the model's output deterministic and focused (minimizing randomness/creativity) — ideal for factual, document-grounded question answering where consistency matters more than variety

### Step 3: Loading and Chunking the PDF

```python
loader = PyPDFLoader("Microsoft-2024-Environmental-Sustainability-Report.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print(f"Success! {len(chunks)} chunks are ready for Groq.")
```

- **`PyPDFLoader(...)`** points to the target PDF file (must already be uploaded into the Colab session's local filesystem).
- **`loader.load()`** parses the PDF and returns a list of LangChain `Document` objects — by default, **one `Document` per page**, each with the page's extracted text as `page_content` and metadata (like page number and source filename).
- **`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)`** configures a text splitter that breaks long text into ~1000-character chunks. It's "recursive" because it tries splitting on natural boundaries first (paragraphs, then sentences, then words) before falling back to a hard character cut, which helps avoid awkwardly severing sentences mid-word. The **100-character overlap** between consecutive chunks helps preserve context that might otherwise be lost right at a chunk boundary — e.g. a sentence that starts at the very end of one chunk and continues into the next.
- **`text_splitter.split_documents(docs)`** applies this splitting across all loaded pages, producing the final `chunks` list — each chunk is itself a `Document`, with its `page_content` capped near 1000 characters.
- The **print statement confirms success** and reports the total chunk count (321 in this run) — a good sanity check that the PDF was parsed and isn't empty.

### Step 4: Asking a Question

```python
context = chunks[100].page_content
query = f"Based on this report snippet, what are the key water findings? Context: {context}"

print("Sending to Groq...")
try:
    response = llm.invoke(query)
    print("\n--- RAG RESULT ---")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
    print("If you still see 'Illegal Header', please double-check your Secrets tab for extra spaces.")
```

- **`chunks[100].page_content`** selects a single, specific chunk (the 101st one, since indexing starts at 0) to use as the retrieval context. This is a **hardcoded, manual retrieval step** — in a full RAG system, this chunk would normally be selected dynamically via a vector similarity search against the user's question, rather than picked by index. Here it's fixed to chunk 100 purely for demonstration.
- **`query`** builds the full prompt sent to the LLM, embedding both an instruction ("what are the key water findings?") and the raw chunk text as `Context:` — this is the core RAG pattern: **ground the model's answer in retrieved text rather than its own training knowledge.**
- **`llm.invoke(query)`** sends the prompt to Groq's API and returns an `AIMessage` object; `.content` extracts the plain-text answer from it.
- The **`try/except`** block wraps the API call so that authentication or network errors are caught and reported clearly, rather than crashing with a raw traceback — and specifically reminds the user to check for the whitespace issue the `clean_key` logic was designed to prevent, in case cleanup wasn't sufficient (e.g. if the key had internal characters causing header issues beyond leading/trailing whitespace).

---

## Output

Running the full script against the Microsoft 2024 Environmental Sustainability Report produces:

```
Success! 321 chunks are ready for Groq.
Sending to Groq...

--- RAG RESULT ---
The key water findings and initiatives mentioned in the report snippet are:

1. **Enhancing safe drinking water availability**: The project aims to ensure the sustainable availability of safe drinking water, particularly in water-starved regions.
2. **Artificial groundwater recharge**: The project will use artificial groundwater recharge to enhance the availability of groundwater in these regions.
3. **Rainwater harvesting**: The project will also harvest rainwater to supplement groundwater supplies.
4. **Climate change adaptation**: The project is designed to help communities adapt to climate change, which is expected to impact water availability in the region.
5. **Scaling up successful initiatives**: The project builds on previous work in Karnataka, which demonstrated significant volumetric benefits, and will be scaled up to other districts in Telangana and North Karnataka.

Overall, the project aims to improve access to safe drinking water in water-scarce regions of India, with a focus on sustainable and climate-resilient solutions.
```

The PDF (321 pages of extracted text, chunked into 321 pieces at ~1000 characters each) was successfully loaded, and the model produced an answer grounded specifically in the content of chunk 100 — a water-related section of the sustainability report — rather than a generic response from its own training data.

---

## Notes & Limitations

- **No real semantic retrieval:** this script does not embed the chunks or perform a similarity search against the query. It always uses `chunks[100]` regardless of what question is asked — for a genuine RAG system, you'd embed all chunks (e.g. with a Groq-compatible or open-source embedding model) into a vector store (like FAISS or Chroma) and retrieve the most relevant chunk(s) per query.
- **Single-chunk context:** only one ~1000-character chunk is used per query. Real-world questions often require synthesizing information across multiple chunks/pages.
- **Colab-specific:** the `userdata.get()` secrets mechanism is specific to Google Colab. Running this outside Colab requires swapping in a different method for loading the API key (e.g. `os.environ.get("GROQ_API_KEY")` with a `.env` file and `python-dotenv`).
- **Hardcoded question:** the query string is fixed in code rather than accepting user input — you'd need to parameterize `query` and `context` selection to make this interactive.
- **`time` import is unused** in the current version of the script.

---

## License

This project is provided as-is for educational/demonstration purposes.
```
