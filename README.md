# 🛡️ Watchdog AI

**AI-Powered Data Quality & Misinformation Detection for Training Datasets**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com/)

Watchdog AI is a comprehensive data curation pipeline designed to clean and validate AI training datasets. It detects misinformation, assesses quality, removes duplicates, and tracks environmental sustainability — all without relying on external LLMs.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Misinformation Detection** | Rule-based pattern matching for clickbait, conspiracy theories, toxic content |
| 📊 **Quality Scoring** | Multi-dimensional scoring: completeness, language quality, information density |
| 🔄 **Duplicate Detection** | Exact (MD5 hash) + Semantic (TF-IDF cosine similarity) duplicate removal |
| 🌍 **Sustainability Tracking** | Carbon footprint calculation via Climatiq API integration |
| 🚀 **REST API** | Production-ready Flask API with CORS support |
| 🖥️ **Web UI** | Interactive frontend for testing and demonstrations |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATASET                            │
│                    (CSV / JSON / JSONL)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Misinformation Detection                               │
│  ├── Suspicious pattern matching (regex)                        │
│  ├── Clickbait phrase detection                                 │
│  ├── Toxicity identification                                    │
│  └── Source credibility scoring                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Quality Assessment                                     │
│  ├── Text completeness (required + optional fields)             │
│  ├── Language quality (capitalization, punctuation)             │
│  ├── Information density (lexical diversity)                    │
│  └── Spam indicator detection                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Duplicate Removal                                      │
│  ├── Exact duplicates (MD5 hashing)                             │
│  └── Semantic duplicates (TF-IDF + Cosine similarity)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Sustainability Impact                                  │
│  ├── Data reduction percentage                                  │
│  ├── Energy savings (kWh)                                       │
│  └── Carbon footprint (kg CO₂) via Climatiq API                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLEANED DATASET                            │
│              + Statistics + Sustainability Report               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Dev-31/WatchdogAI.git
cd WatchdogAI
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your CLIMATIQ_API_KEY (optional)
```

### 3. Run the API Server

```bash
python api/app.py
```

API will be available at: `http://localhost:5000`

### 4. Open the UI

Open `frontend.html` in your browser to access the interactive interface.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze single text for misinformation + quality |
| `/analyze/batch` | POST | Batch analysis of multiple texts |
| `/quality` | POST | Quality scoring only |
| `/duplicates` | POST | Find duplicates in text list |
| `/process` | POST | Full 4-step pipeline processing |
| `/sustainability` | POST | Calculate carbon savings |

### Example: Analyze Text

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "source": "example.com"}'
```

**Response:**
```json
{
  "status": "clean",
  "misinformation_score": 0.12,
  "confidence": 0.45,
  "risk_level": "low",
  "quality_score": 0.82,
  "quality_level": "high",
  "flags": [],
  "explanations": ["No significant misinformation indicators detected"]
}
```

---

## 🔧 Core Modules

### MisinformationDetector
Detects problematic content using regex pattern matching:
- **Suspicious patterns**: Conspiracy theories, miracle cures, sensationalism
- **Clickbait phrases**: "You won't believe", "Number X will shock you"
- **Toxicity**: Hate speech, insults, aggressive language
- **Source credibility**: Boosts `.gov`, `.edu`; penalizes suspicious domains

### DataQualityScorer
Evaluates structural quality with weighted scoring:
- Information density: **28%**
- Language quality: **25%**
- Word count: **13%**
- Completeness: **12%**
- Spam check: **12%**
- Text length: **10%**

### RedundancyDetector
Two-stage duplicate detection:
1. **Exact**: MD5 hash-based O(1) lookup
2. **Semantic**: TF-IDF vectorization + cosine similarity (threshold: 85%)

### SustainabilityTracker
Environmental impact calculation:
- Integrates with **Climatiq API** for real carbon data
- Fallback to regional averages (Global: 0.475 kg CO₂/kWh)
- Tracks energy, carbon, and water usage

---

## 📋 Detection Triggers Reference

| Type | Examples |
|------|----------|
| Clickbait | "you won't believe", "shocking", "number X will..." |
| Conspiracy | "deep state", "illuminati", "wake up sheeple" |
| Toxicity | "idiot", "stupid", "hate", "disaster", "fraud" |
| Spam | "FREE", "CLICK HERE", "GUARANTEED", "ACT NOW" |
| Low Quality | Excessive filler words: "good", "stuff", "very" |
| Excessive Caps | >40% uppercase letters |
| Excessive Punctuation | Multiple `!!!` or `???` |

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Technical Walkthrough (PDF)](Watchdog_AI_Technical_Walkthrough.pdf) | Complete code explanation for every function |
| [Demo Inputs (PDF)](Watchdog_AI_Demo_Inputs.pdf) | 10 test cases for live demonstrations |

---

## 🧪 Testing

```bash
# Run quick check
python quick_check.py

# Run verification checkpoints
python verify_checkpoints.py

# Run tests
pytest tests/
```

---

## 📁 Project Structure

```
WatchdogAI/
├── api/
│   └── app.py              # Flask REST API
├── src/
│   ├── misinformation_detector.py
│   ├── quality_scorer.py
│   ├── redundancy_detector.py
│   ├── sustainability_tracker.py
│   └── dataset_processor.py
├── docs/
│   ├── generate_walkthrough_pdf.py
│   └── generate_demo_inputs_pdf.py
├── frontend.html           # Web UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🌍 Sustainability Impact

When you clean your dataset with Watchdog AI, you receive:
- **Immediate savings**: Data reduction %, energy saved (kWh), carbon saved (kg CO₂)
- **Annual projections**: Extrapolated yearly impact
- **Equivalencies**: Trees planted, car miles avoided

---

## ⚙️ Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `CLIMATIQ_API_KEY` | API key for carbon calculations | Optional |

Without the API key, sustainability tracking uses regional fallback values.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Climatiq](https://www.climatiq.io/) for carbon emissions API
- [scikit-learn](https://scikit-learn.org/) for TF-IDF vectorization
- [Flask](https://flask.palletsprojects.com/) for the REST API framework

---

<p align="center">
  Built with ❤️ for cleaner AI training data
</p>
