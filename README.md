# Signal Foundry  
### The Unstructured Data Intel Engine  
**A b*llshit-seeking missile disguised as a text analytics tool.**

Most "intelligence" tools drown you in **how** — elaborate mechanisms, 400-page reports, 47-step frameworks, and 19 new acronyms that all mean "we're doing something." (aka "mechanistic porn")

Signal Foundry was built for one purpose: **rip away the "how" and expose the "why".**

In <60 seconds it shows you:
- Who actually matters (entities)
- What they’re really talking about (NPMI sticky phrases)
- What they’re deliberately trying to hide (network graph clusters that refuse to connect)

Because the **why** always leaks through the terminology, the funding sources, and the topological shape of the language — never through the official narrative.

Current Version: v2.9 — Production, Air-Gap Ready, Graphics-Hardened

### Core Thesis
**Every act of motivated obfuscation leaves a mathematical fingerprint - and this's true for...accidental obfuscation too~.**  
Signal Foundry is designed from the ground up to detect that fingerprint and make it hard to ignore.

### Key Capabilities

**1. Privacy-First & Air-Gap Native**  
- Zero data egress by default  
- Offline "Harvester" mode (`harvester.py`) for b|ack-site / SC|F use  
- Raw text is discarded immediately after statistical sketching  
- Optional AI Analyst sees only frequencies and correlations — never the source

**2. The How → Why Arsenal**

| Feature                  | What It Actually Reveals                                      |
|--------------------------|----------------------------------------------------------------|
| NPMI Phrases             | The real "terms of art" that betray intent ("dual use", "cognitive liberty", "chain of custody") |
| Polymorphic NER          | Who benefits, who funds, who executes |
| Network Graph            | Topological proof of separate intent clusters in the same document ("performance enhancement" vs "interrogation") |
| TF-IDF Keyphrases        | Unique technical DNA that survives boilerplate removal |
| Entities + Graph together| Reveals: "Whose interest is actually being served here?" |

**3. Hybrid Architecture**

- Interactive Mode (`mainapp.py`): Drag-drop Streamlit interface for instant insight
- Headless Harvester (harvester.py): Process 10M+ row datasets on secure servers. Generates a lightweight statistical sketch (NPMI/Counts/Graph only) to maximize privacy and speed
- Additive Scanning: Merge multiple files/leaks into a single intelligence picture without resetting

**4. Provenance & Chain of Custody**

Hybrid Signature: A scannable QR-overlaid heatmap that cryptographically binds the visualization to the exact dataset hash.  
Proof that your insight came from *this* document, not an AI hallucination.

### Real-World Use Cases (Where It Hurts Them)

- Mapping hidden intent in defense/neurotech policy documents
- Detecting when "safety" language is being weaponized as cover
- Identifying which three entities keep appearing together across 400 pages
- Seeing that "non-invasive" and "transcranial" are the real threat vector, not sci-fi implants
- Literary/digital forensics — instantly spotting ghostwriting or translated artifacts
- Crisis timeline reconstruction — watching "leak" and "failure" spike three weeks before the public story breaks

### Installation

```bash
git clone https://github.com/yourusername/signal-foundry.git
cd signal-foundry
pip install -r requirements.txt


Create .streamlit/secrets.toml:

toml

auth_password = "your-secure-password-here"

# Optional AI (still never sees the RAW text)
# openai_api_key = "sk-..."
# xai_api_key = "..."

Run:

Bash

# Interactive mode
streamlit run mainapp.py

# Air-gapped / large-scale mode
python harvester.py --input "classified_dataset.csv" --col "text" --output "sketch.json"

Supported Formats

CSV, Excel, PDF, PowerPoint, VTT, JSON/JSONL, URLs, raw text — whatever you’ve got.
Dependencies (Graceful Degradation)

Works with just pandas + streamlit.
Install more → unlock graphs, topic modeling, sentiment, QR signatures.
License

License: MIT
Status

Production v2.9
Used in real investigations.
May cause multiple "wait-what!" moments in under two minutes~
