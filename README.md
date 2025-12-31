Signal Foundry: The Unstructured Data Intelligence Engine

Signal Foundry is a lightweight, privacy-first analytics platform designed to extract mathematical structure, temporal trends, and qualitative insights from massive, messy text datasets ("dirty data").

Unlike traditional NLP tools that require loading entire datasets into RAM, Signal Foundry uses a Streaming + Sketching architecture. It reads files in small chunks, extracts statistical summaries (n-grams, entity counts, vocabulary distributions) into a lightweight "Sketch," and discards the raw text immediately. This allows it to process gigabytes of data on standard laptops with minimal memory footprint.

Current Version: v2.9 (Robust Graphics Safety)
🚀 Key Features
1. 🛡️ Privacy-First & Air-Gap Ready

    Zero Data Egress: All core processing happens locally. No data is sent to the cloud unless you explicitly enable the optional "AI Analyst" feature.
    The "Privacy Proxy": Includes a Data Refinery tool to strip PII, chat logs, and HTML from datasets before sharing them with third-party tools.

2. ⚡ Hybrid Architecture

    Interactive Mode (Streamlit): Drag-and-drop interface for immediate visual analysis.
    Headless Mode (Harvester): A CLI utility (harvester.py) for processing massive files (10M+ rows) on remote servers without a GUI.
    Additive Scanning: Pause, resume, and merge scans from multiple files into a single intelligence picture.

3. 🧠 Advanced Analytics

    Network Graphing: Physics-based visualization to see how concepts link together (e.g., clustering "Performance" separately from "Interrogation" in the same document).
    Polymorphic NER: Heuristic-based Named Entity Recognition that identifies Acronyms (DARPA), IDs (COVID-19), and Proper Names without heavy ML models.
    Temporal Analysis: Automatically builds time-series charts to show volume and term trends if a date column is detected.
    Hybrid Signature: Generates a cryptographic QR-coded heatmap to prove the "Chain of Custody" for your analysis.

4. 🤖 Optional AI Analyst

    Integrates with OpenAI (GPT-4o) or xAI (Grok).
    Privacy Guard: The AI only sees the statistical metadata (frequencies and correlations), never the raw document text.

📖 Use Cases

    Crisis Timeline Reconstruction: Map high-severity words (e.g., "leak", "fail") over time to pinpoint exactly when an incident started.
    Stakeholder Mapping: Instantly see Who and What are driving conversations using Entity Detection.
    Literary Forensics: Analyze vocabulary diversity and phrase patterns to detect authorship style or "ghostwriting."
    LMS Forum Analysis: Visualize student sentiment trends and "Unknown Unknowns" (topics students are confused about but aren't asking directly).

🛠️ Installation & Setup
1. Clone & Install

Bash

git clone https://github.com/yourusername/signal-foundry.git
cd signal-foundry
pip install -r requirements.txt

2. Configuration (Crucial Step)

The app requires a password for the login screen and optional API keys for AI features.
Create a file at .streamlit/secrets.toml:

toml

# .streamlit/secrets.toml

auth_password = "your-secure-password"

# Optional: For AI Analyst features
openai_api_key = "sk-..."
xai_api_key = "..."

3. Running the App

Interactive Mode (The Viewer):

Bash

streamlit run mainapp.py

On first run, the app will automatically check for and download necessary NLTK corpora (WordNet, VADER).

Headless Mode (The Harvester):
For processing massive CSVs/Excels on a server without a display:

Bash

python harvester.py --input "huge_dataset.csv" --col "message_text" --output "my_sketch.pkl"

📊 Supported Formats

Signal Foundry ingests specific text columns or raw content from:

    CSV / Excel (.xlsx): Auto-detects headers; allows specific column selection for Text, Date, and Category.
    PDF: Extracts raw text layer (requires pypdf).
    PowerPoint (.pptx): Extracts slide text (requires python-pptx).
    VTT: Video transcripts (automatically strips timestamps).
    JSON / JSONL: Stream processing for logs.
    URLs: Basic web scraping for quick intel gathering.

🧩 Dependencies

The engine is built on a "Soft Dependency" model. It will run with minimal requirements, but features unlock as you install more packages:

    streamlit, pandas, numpy, matplotlib (Core)
    networkx (Graphing)
    scikit-learn (Topic Modeling)
    nltk (Sentiment & Lemmatization)
    openpyxl, pypdf, python-pptx (File Readers)
    qrcode (Hybrid Signatures)

License: MIT
Status: Production (v2.9)
