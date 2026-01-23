# Signal Foundry  
### The Unstructured Data Intel Engine  
**A b*llshit-seeking missile disguised as a text analytics tool.**

Most "intelligence" tools drown you in **how** — elaborate mechanisms, 400-page reports, 47-step frameworks, and 19 new acronyms that all mean "we're doing something." (aka "mechanistic porn")

Signal Foundry was built for one purpose: **rip away the "how" and expose the "why".**

In <60 seconds it shows you:
- **Who** actually matters (Entities & Stakeholders)
- **What** they’re really talking about (NPMI sticky phrases)
- **Where** they sit on the evolutionary scale (Maturity Modeling)
- **Why** they are saying it (Network graph clusters that refuse to connect)

Because the **why** always leaks through the terminology, the funding sources, and the topological shape of the language — never through the official narrative.

**Current Version:** v2.9 — Production, Air-Gap Ready, Graphics-Hardened

### Core Thesis
**Every act of motivated obfuscation leaves a mathematical fingerprint — and this is true for accidental incompetence too.**  
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
| **Maturity Engine**      | **NEW:** Scores organizational capability (1.0–5.0) using linguistic personas (EdTech vs. Policy vs. Corp). |
| NPMI Phrases             | The real "terms of art" that betray intent ("dual use", "cognitive liberty", "chain of custody"). |
| Polymorphic NER          | Who benefits, who funds, who executes. |
| Network Graph            | Topological proof of separate intent clusters in the same document ("performance enhancement" vs "interrogation"). |
| TF-IDF Keyphrases        | Unique technical DNA that survives boilerplate removal. |
| Entities + Graph         | Reveals: "Whose interest is actually being served here?" |

**3. Multi-Persona Architecture**

Context changes meaning. The word *"Risk"* means one thing in a University (At-Risk Students) and another in the UN (Security Threat).
- **Selectable Personas:** Switch instantly between **EdTech/LMS Ops**, **General Business**, and **Policy & Governance** models to score documents against the correct framework.

**4. Provenance & Chain of Custody**

- **Hybrid Signature:** A scannable QR-overlaid heatmap that cryptographically binds the visualization to the exact dataset hash. Proof that your insight came from *this* document, not an AI hallucination.

### Real-World Use Cases (Where It Hurts Them)

- **The Support Audit:** Scanning 5,000 LMS support tickets to prove an institution is stuck in "Reactive/L1" hell despite claiming "Strategic/L5" vision.
- **Crisis Reconstruction:** Watching "leak" and "failure" spike three weeks before the public story breaks via the Time-Travel slider.
- **Policy decoding:** Identifying which three entities keep appearing together across 400 pages of defense policy.
- **Strategic Alignment:** Scanning the CTO's emails vs. the CEO's annual letter to spot execution gaps.
- **Forensics:** Seeing that "non-invasive" and "transcranial" are the real threat vector, not sci-fi implants.

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

CSV, Excel, PDF, PowerPoint, VTT (Transcripts), JSON/JSONL, URLs, raw text.
Dependencies (Graceful Degradation)

Works with just pandas + streamlit.
Install more → unlock graphs, topic modeling, sentiment, QR signatures.
License

License: MIT
Status

Production v2.9
Used in real investigations.
May cause multiple "wait-what!" moments in under two minutes~

Signal Foundry is a linguistic MRI machine that ignores the official narrative to reveal the skeletal structure of intent, maturity, and risk hidden within your data
