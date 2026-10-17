# IT Assist AI — AI-Powered IT Help Desk Chatbot

## 1. Project Overview

**IT Assist AI** is an AI-powered IT help desk chatbot that provides troubleshooting guidance using an organization's IT knowledge base. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from approved IT documents and generate helpful responses to user questions.

The chatbot helps users troubleshoot common IT issues such as Wi-Fi connectivity, VPN errors, locked accounts, slow computers, and printer problems.

## 2. Objectives

- Provide quick answers to common IT support questions.
- Retrieve relevant information from IT troubleshooting documents.
- Generate answers based on available knowledge.
- Display supporting document passages for verification.
- Avoid generating unsupported answers when information is unavailable.
- Provide a simple, user-friendly interface for IT support.

## 3. Key Features

- **Document ingestion:** Supports PDF and TXT troubleshooting documents.
- **Semantic search:** Finds relevant passages using text embeddings.
- **Vector similarity search:** Uses FAISS to retrieve relevant document chunks.
- **AI-powered responses:** Uses a locally running Ollama language model.
- **Source references:** Allows users to inspect supporting passages.
- **Conversation history:** Displays questions and answers during the session.
- **Fallback responses:** Indicates when the knowledge base lacks sufficient information.
- **Privacy reminder:** Warns users not to submit passwords, API keys, or confidential information without approval.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Sentence Transformers | Text embedding generation |
| FAISS | Vector similarity search |
| Ollama | Local language model execution |
| Qwen2.5-0.5B | Language model for answer generation |
| PyPDF | PDF text extraction |
| NumPy | Numerical operations |

## 5. System Architecture

The application follows a Retrieval-Augmented Generation (RAG) workflow.

1. **Document loading:** Reads approved IT troubleshooting documents.
2. **Text extraction:** Extracts text from PDF and TXT files.
3. **Text chunking:** Divides documents into smaller passages.
4. **Embedding generation:** Converts text chunks into numerical vectors.
5. **Index creation:** Stores vectors in a FAISS similarity-search index.
6. **Question processing:** Converts the user's question into an embedding.
7. **Information retrieval:** Retrieves relevant passages from the knowledge base.
8. **Answer generation:** Uses the retrieved context to generate a response.
9. **Source inspection:** Displays supporting passages for user verification.

## 6. Project Structure

```text
IT-Assist-AI/
│
├── app.py
├── rag_engine.py
├── documents/
│   └── it_troubleshooting.txt
├── requirements.txt
└── README.md
```

The project may contain additional files or folders depending on your implementation.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Visual Studio Code
- Ollama
- An internet connection for initial Python package and embedding-model downloads

### Step 1: Create a virtual environment

Open the VS Code terminal in your project folder and run:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### Step 2: Install dependencies

Create a `requirements.txt` file with:

```text
streamlit
pypdf
sentence-transformers
faiss-cpu
requests
numpy
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

### Step 3: Set up Ollama

Install Ollama and download the required language model:

```bash
ollama pull qwen2.5:0.5b
```

Ensure Ollama is running locally and the model is available before testing answer generation.

### Step 4: Add the knowledge base

Place your approved troubleshooting document inside the `documents` folder.

Example:

```text
documents/it_troubleshooting.txt
```

### Step 5: Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Build or refresh the knowledge base from the application's sidebar before asking questions.

## 8. Sample Test Cases

| Test question | Expected behavior | Result observed |
|---|---|---|
| What information should I record if my VPN fails? | Explain that the exact error message should be recorded and reported if the problem persists. | Passed |
| My Wi-Fi is not connecting. What should I check? | Provide relevant Wi-Fi troubleshooting steps. | Passed |
| My computer is running slowly. What should I do? | Suggest checking resource usage, storage, and unnecessary applications. | Passed |
| My account is locked. How can I access it? | Recommend contacting an authorized administrator. | Passed |
| What is the company's travel reimbursement limit? | Indicate that the available documents do not contain sufficient information. | Passed |

*These results are based on initial manual testing. They do not represent a comprehensive evaluation of the system.*

## 9. Advantages

- Reduces the effort required to find basic IT troubleshooting instructions.
- Provides answers using an organization-specific knowledge base.
- Supports inspection of retrieved document passages.
- Can run the language model locally through Ollama.
- Can be extended with additional approved troubleshooting documents.
- Offers a simple interface for non-technical users.

## 10. Limitations

- Answer quality depends on the relevance and quality of the available documents.
- The language model may occasionally generate incomplete or incorrect responses.
- Scanned PDFs may require OCR if they do not contain extractable text.
- The prototype does not automatically connect to an organization's ticketing system.
- Knowledge-base indexing and persistence depend on the current implementation.
- The application is not a replacement for authorized IT support or organizational security policies.

## 11. Future Enhancements

- Add a larger collection of approved IT support documents.
- Improve retrieval accuracy and relevance ranking.
- Add automated evaluation using a larger set of test questions.
- Save and restore the vector index between application sessions.
- Integrate with IT service management platforms.
- Add user authentication and access control where required.
- Support multilingual troubleshooting responses.
- Deploy the application in an approved internal environment.

## 12. Security and Responsible Use

- Never enter passwords, API keys, access tokens, or personal secrets into the chatbot.
- Upload only documents approved for use with the application.
- Verify important instructions with authorized IT support.
- Do not treat generated responses as authoritative security or access-control decisions.

## 13. Conclusion

IT Assist AI demonstrates how Retrieval-Augmented Generation can be used to build a knowledge-powered IT help desk chatbot. By combining document retrieval, semantic search, and local language-model generation, the prototype provides accessible troubleshooting guidance and indicates when the available knowledge base does not support an answer.

**Project:** IT Assist AI — AI-Powered IT Help Desk Chatbot  
**Category:** Artificial Intelligence / Natural Language Processing  
**Core approach:** Retrieval-Augmented Generation (RAG)  
**Interface:** Streamlit  
**Language model runtime:** Ollama
