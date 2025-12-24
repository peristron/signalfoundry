Signal Foundry: The Unstructured Data Intelligence Engine

Signal Foundry is a lightweight, privacy-first analytics platform designed to extract mathematical structure, temporal trends, and qualitative insights from massive, messy text datasets ("dirty data").

Unlike traditional NLP tools that require loading entire datasets into RAM, Signal Foundry uses a Streaming + Sketching architecture. It reads files in small chunks, extracts statistical summaries (n-grams, entity counts, vocabulary distributions) into a lightweight "Sketch," and discards the raw text immediately. This allows it to process gigabytes of data on standard laptops with minimal memory footprint.
🚀 Key Features
1. 🛡️ Privacy-First & Offline

    Zero Data Egress: All processing happens locally in your browser/server. No data is sent to external APIs (unless you explicitly use the AI Analyst feature).
    The "Privacy Proxy": Use the built-in Data Refinery to strip PII, chat logs, and HTML from datasets before sharing them with third-party tools or LLMs.

2. ⚡ Streaming Architecture ("The Sketch")

    Memory Safe: Processes 10M+ rows as easily as 100 rows.
    Stateful Analysis: Pause, resume, and merge scans from multiple files (e.g., combining 12 months of CSV logs into a single analysis).

3. 🧠 Advanced NLP (Batteries Included)

    Lemmatization: Intelligent merging of word forms (run, running, ran → run) for accurate counting.
    NER Lite: Heuristic-based Named Entity Recognition to identify People, Organizations, and Products without heavy ML models.
    Temporal Analysis: If a date column is provided, the engine automatically builds time-series charts to show volume and term trends over time.
    Keyphrase Scoring (TF-IDF): Identifies words that are statistically unique to specific documents, filtering out high-frequency noise.

📖 Use Cases
🏢 Corporate & Strategic

    Crisis Timeline Reconstruction: Map high-severity words (e.g., "leak", "fail") over time to pinpoint exactly when an incident started.
    Stakeholder Mapping: Instantly see Who and What are driving conversations using Entity Detection.
    M&A Due Diligence: Rapidly scan data rooms for liability terms (lawsuits, risks) without manually reading thousands of documents.

🔬 Research & Forensics

    "Year-in-Review" Retrospectives: Ingest a full year of journals or logs to visualize how themes shifted from Q1 to Q4.
    Signal vs. Noise: Use TF-IDF extraction to ignore generic corporate jargon and focus on the unique identifiers of a project or team.
    Literary Forensics: Analyze vocabulary diversity and phrase patterns to detect authorship style or "ghostwriting."

🎓 Education & Learning

    LMS Forum Analysis: visualize student sentiment trends before and after exams.
    Curriculum Audit: Identify the most frequently discussed concepts vs. the "Unknown Unknowns" (topics students are confused about but aren't asking directly).

🛠️ Installation & Setup

    Clone the repository:

Bash

git clone https://github.com/yourusername/signal-foundry.git
cd signal-foundry

Install requirements:

Bash

pip install -r requirements.txt

Run the App:

Bash

    streamlit run mainapp.py

    First Run:
        The app will automatically download necessary NLTK corpora (WordNet, VADER, etc.) on the first launch.

📊 supported Formats

Signal Foundry ingests specific text columns or raw content from:

    CSV / Excel (.xlsx): (Auto-detects headers and allows column selection)
    PDF: (Extracts raw text layer)
    JSON / JSONL: (Stream processing for logs)
    VTT: (Video transcripts)
    PowerPoint (.pptx): (Extracts slide text)

License: MIT
Status: Production (v2.0)
