# 📚 Research Paper Assistant

An AI-powered Research Paper Assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded research papers.

The system allows users to upload one or more PDF research papers, converts their content into searchable vector representations, retrieves the most relevant passages for a question, and uses a Large Language Model (LLM) to generate an evidence-grounded answer with source and page references.

---

## 🚀 Live Demo

**Live Application:**  
Add your deployed Streamlit URL here.

Example:

https://your-app-name.streamlit.app

---

## 📌 Project Overview

Reading and understanding multiple research papers can be time-consuming, especially when users need to find specific information across several documents.

This project provides a simple AI-based solution using Retrieval-Augmented Generation.

Instead of asking an LLM to answer a question using only its pretrained knowledge, the system first searches the uploaded research papers and provides the most relevant passages to the LLM.

The LLM then generates an answer based on the retrieved evidence.

### Basic workflow

```text
Research Papers
       ↓
     PDF
       ↓
 Text Extraction
       ↓
    Chunking
       ↓
   Embeddings
       ↓
   ChromaDB
       ↓
 User Question
       ↓
 Semantic Retrieval
       ↓
 Relevant Passages
       ↓
      LLM
       ↓
 Answer + Sources

✨ Features
    📄 Upload research papers in PDF format
    📚 Support for multiple research papers
    ✂️ Automatic text chunking
    🧠 Local semantic embeddings
    🔎 Semantic similarity search
    🗃️ Persistent ChromaDB vector database
    🤖 LLM-based answer generation
    🔗 Source and page references
    📑 Multi-document question answering
    🛡️ Evidence-grounded prompting
    🚫 Reduced unsupported/hallucinated answers
    🔄 LLM provider fallback
    🌐 Streamlit web interface
    💰 Designed to use free/low-cost API tiers

🏗️ System Architecture
                  ┌──────────────────────┐
                  │   Research Papers    │
                  │       (PDFs)         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   PDF Text           │
                  │   Extraction         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Chunking        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Sentence Transformer │
                  │    Embeddings        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      ChromaDB        │
                  │   Vector Database    │
                  └──────────┬───────────┘
                             │
                             │
User Question ────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Semantic Retrieval   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Relevant Paper       │
                  │ Passages             │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │        LLM           │
                  │    Generation       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Answer + Sources     │
                  │ + Page References    │
                  └──────────────────────┘

🧰 Technology Stack
| Component            | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Web Interface        | Streamlit                  |
| PDF Processing       | PyPDF                      |
| Embeddings           | Sentence Transformers      |
| Embedding Model      | all-MiniLM-L6-v2           |
| Vector Database      | ChromaDB                   |
| LLM Providers        | Gemini / Groq / OpenRouter |
| HTTP Requests        | Requests                   |
| Configuration        | python-dotenv              |

📁 Project Structure
research-paper-assistant/
│
├── app.py
│   └── Streamlit user interface
│
├── rag.py
│   └── PDF processing, chunking,
│       embeddings, retrieval and prompting
│
├── llm.py
│   └── LLM API calls and provider fallback
│
├── config.py
│   └── Application configuration
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Files excluded from Git
│
├── documents/
│   └── Optional local PDF storage
│
└── data/
    └── chroma/
        └── ChromaDB data

🔍 How RAG Works in This Project

Retrieval-Augmented Generation combines information retrieval with Large Language Model generation.

The system follows two major stages.

1. Retrieval

When a PDF is uploaded:

Text is extracted from the PDF.
The text is divided into smaller chunks.
Each chunk is converted into an embedding vector.
The embeddings are stored in ChromaDB.

When the user asks a question:

The question is converted into an embedding.
ChromaDB searches for semantically similar chunks.
The most relevant chunks are retrieved.
The retrieved chunks are supplied to the LLM.
2. Generation

The LLM receives:

User Question
+
Retrieved Paper Evidence

It is instructed to answer using the retrieved evidence rather than relying on unsupported information.

The application also displays the retrieved document and page information so that users can inspect the supporting evidence.

🧠 Embedding Model

The project uses:

all-MiniLM-L6-v2

through Sentence Transformers.

The embedding model converts text into numerical vectors that represent semantic meaning.

For example:

"What causes cascading cloud incidents?"

and:

"Multiple anomalies can originate from a single underlying failure."

may have similar vector representations even though the words are different.

This allows semantic retrieval rather than simple keyword matching.

🗃️ Vector Database

The project uses ChromaDB.

ChromaDB stores:

Text chunks
Embeddings
PDF filenames
Page numbers
Chunk information

Example metadata:

filename:
CausalDX_Diagnosing_Long-Tail_and_Cascading_Cloud_Incidents_With_LLM-Guided_Causal_Reasoning.pdf

page:
10

chunk:
2

This metadata allows the application to show source information with the generated answer.

🤖 LLM Providers

The application supports multiple LLM providers.

The current fallback architecture is:

Gemini
   ↓
Groq
   ↓
OpenRouter

If a configured provider fails, the application attempts the next configured provider.

This can help handle temporary provider availability problems or rate limits.

The project is designed so that providers can be changed through environment variables without modifying the main RAG pipeline.

🔐 Environment Variables

Create a .env file locally.

Example:

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_model

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free

Never commit .env to GitHub.

⚙️ Installation
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL

Move into the project:

cd research-paper-assistant
2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure API keys

Create:

.env

and add the required API keys and model names.

5. Run the application
streamlit run app.py

The application will normally be available at:

http://localhost:8501
📖 How to Use
Step 1 — Upload Papers

Open the application and upload one or more PDF research papers.

Step 2 — Index Papers

Click:

Index Papers

The application extracts the text, creates chunks, generates embeddings, and stores them in ChromaDB.

Step 3 — Ask a Question

Enter a question such as:

What is the main problem addressed by this paper?

Click:

Ask
Step 4 — Inspect Sources

The application displays the retrieved passages together with:

PDF filename
Page number
Retrieved text

This allows the user to inspect the evidence used to generate the answer.

🧪 Example Questions
Single-paper questions
What is the main problem addressed by this paper?
What methodology does the paper propose?
What are the main contributions of the paper?
Technical questions
What is AGRCS?
How does the proposed system perform root cause analysis?
What evaluation metrics are reported?
Multi-paper questions
Compare the approaches proposed in the two uploaded papers.
What similarities exist between the two papers?
What are the main differences between the two approaches?
Evidence test
What programming language was used to implement the system?

If the information is not present in the uploaded documents, the system should indicate that the information cannot be determined from the available evidence.

🧪 RAG Testing

The system can be tested at several levels.

Retrieval Test

Ask a question whose answer clearly exists in a paper.

Check whether the retrieved source contains the relevant passage.

Multi-document Test

Upload two papers and ask:

Compare the main approaches proposed in the two papers.

Check whether relevant chunks from both papers are retrieved.

Evidence Test

Ask a question whose answer is not present in the uploaded papers.

The system should avoid inventing unsupported information.

Provider Test

Configure multiple LLM providers and temporarily disable one provider.

The fallback system should attempt the next configured provider.

🛡️ Evidence-Grounded Answering

The application instructs the LLM to:

Use retrieved paper content as evidence.
Avoid unsupported claims.
Avoid using external knowledge for the answer.
Identify when the retrieved context is insufficient.
Provide source and page information where possible.

This approach helps reduce unsupported answers and makes the generated responses easier to verify.

☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

Basic deployment process
Push the project to GitHub.
Open Streamlit Community Cloud.
Connect the GitHub repository.
Select the repository.
Select the main branch.
Select:
app.py

as the main application file.
7. Configure API keys through the deployment platform's secrets/settings.
8. Deploy the application.

After deployment, the application receives a public URL.

Example:

https://research-paper-assistant.streamlit.app
🔒 Security

API keys must never be committed to GitHub.

The following files and directories should remain private:

.env
venv/
.venv/
data/chroma/
__pycache__/

The .gitignore file should include:

.env
venv/
.venv/
data/chroma/
__pycache__/
*.pyc

If an API key is accidentally exposed publicly, revoke or rotate it immediately.

⚠️ Limitations

The current version has several limitations.

Scanned PDFs

PDFs containing only scanned images may not contain machine-readable text.

OCR support can be added in a future version.

Chunking

The current implementation uses relatively simple text chunking.

More advanced section-aware or paragraph-aware chunking could improve retrieval.

Retrieval

Semantic retrieval can sometimes return passages that are related to the question but are not the most useful evidence.

A future version could use:

Hybrid search
Reranking
Metadata filtering
Query expansion
LLM Availability

API availability, rate limits, model names, and free-tier policies can change between providers.

The application therefore uses a provider fallback mechanism.

Persistent Storage

The current local ChromaDB setup is suitable for development and demonstrations.

A production deployment may require a managed or persistent database depending on the deployment environment.

🔮 Future Improvements

Possible future improvements include:

💬 Conversation history
📚 Improved multi-paper comparison
🔎 Hybrid keyword + semantic search
🎯 Retrieval reranking
🧩 Section-aware document chunking
📌 More detailed citations
📊 RAG evaluation metrics
🧪 Automated retrieval evaluation
📈 Answer quality evaluation
📝 Paper summarization
🔬 Research methodology extraction
📑 Automatic comparison tables
🔐 Improved document isolation
🗂️ User-specific document collections
🎓 Project Learning Outcomes

This project demonstrates practical understanding of:

Retrieval-Augmented Generation
Large Language Models
Vector embeddings
Semantic search
Vector databases
PDF document processing
Prompt engineering
Multi-document retrieval
API integration
LLM fallback mechanisms
Streamlit application development
Evidence-grounded AI systems
📌 Project Status
Version: 1.0

Status: Working Prototype

Implemented:

 PDF upload
 PDF text extraction
 Text chunking
 Embedding generation
 ChromaDB storage
 Semantic retrieval
 Multi-document retrieval
 LLM answer generation
 Source/page references
 Evidence-grounded prompting
 LLM provider fallback
 Streamlit interface
👨‍💻 Author

Your Name

Computer Science / Artificial Intelligence Student

📜 License

This project is intended for educational and research purposes.