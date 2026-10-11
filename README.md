# AI Research Assistant using Mini RAG




## Files

| File | Job |
|---|---|
| `app.py` | Streamlit user interface (run this one) |
| `rag_engine.py` | PDF reading, chunking, embeddings, FAISS, retrieval |
| `llm.py` | Hosted Groq LLM API (no local text-generation model) |
| `features.py` | Q&A, summary, gaps, compare, notes, quiz, mentor |
| `citations.py` | APA 7th and IEEE references and in-text citations |
| `requirements.txt` | Libraries to install |

## Setup (Windows, VS Code)

1. Install **Python 3.10, 3.11 or 3.12** (tick "Add Python to PATH"). Avoid 3.13+ for now; some libraries lag behind.
2. Open this folder in VS Code, then open **Terminal > New Terminal**.
3. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   (Mac/Linux: `source venv/bin/activate`)
4. Install everything (needs internet, takes 5 to 15 minutes):
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run:
   ```
   streamlit run app.py
   ```
   The browser opens at http://localhost:8501

The first run downloads the sentence-embedding model (about 90 MB). Answer generation uses the hosted Groq API; no local text-generation model is downloaded or run. PDF embeddings and FAISS retrieval still run on your computer.

## Hosted answer generation with Groq

1. Create an account at https://console.groq.com/ and create an API key.
2. Run the app and paste the key in the sidebar under **Hosted LLM settings**.
3. Keep the key private and do not paste it into source code or share screenshots containing it.

The default model is `llama-3.1-8b-instant`. API availability, model access, quotas and speed depend on the provider and your account. If the API fails, the app shows an error instead of starting a slow local language model.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` | Activate the venv first (step 3), or use `python -m streamlit run app.py` |
| `No module named ...` | Run `pip install -r requirements.txt` inside the activated venv |
| `faiss` fails to install | Use Python 3.10 to 3.12, then `pip install faiss-cpu` |
| "No readable text found" | The PDF is a scan. Use a PDF where you can select text with the mouse |
| Groq API error | Check the API key, internet connection, model ID, and provider usage limits. The app does not run a local LLM fallback. |
| First PDF upload is slow | The embedding model may be downloading or the PDF is being indexed. Later uploads of the same paper during the session can reuse the index. |
