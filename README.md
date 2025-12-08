# Surabaya-Opinion-Analytics

**Surabaya Public Opinion Analyzer: Topic Modeling, Sentiment, Location Extraction and Mapping of Urban Issues on Social Media**

A comprehensive NLP pipeline for analyzing public opinions about Surabaya from social media (X/Twitter and Reddit), featuring automated topic discovery, sentiment analysis, and location extraction with weekly data updates and monthly model retraining.

[![Dashboard](https://img.shields.io/badge/🤗%20Hugging%20Face-Dashboard-yellow)](https://huggingface.co/spaces/Kiuyha/Surabaya-Opinion-Analytics)
[![Topic Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Topic%20Model-blue)](https://huggingface.co/Kiuyha/surabaya-opinion-tweet-clusters)
[![NER Model](https://img.shields.io/badge/🤗%20Hugging%20Face-NER%20Model-green)](https://huggingface.co/Kiuyha/surabaya-opinion-indobert-ner)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-NER%20Dataset-orange)](https://huggingface.co/datasets/Kiuyha/surabaya-ner-dataset)

## Features

### 🔍 Data Collection
- **Multi-source scraping**: Custom Nitter scraper for X/Twitter and API integration with Reddit via `api.pullpush.io`
- **Automated updates**: GitHub Actions workflow runs weekly to collect fresh data
- **Cloud storage**: Supabase integration for persistent data storage and real-time dashboard updates

### 📊 Topic Modeling
Our topic modeling pipeline uses an optimized approach combining FastText embeddings and K-Means clustering:

- **Optimal K selection** via composite scoring:
  ```
  Final Score = 0.6 × Coherence (C_v) + 0.4 × Separation (Cosine Distance)
  ```
- **Quality filtering**: Automatically removes low-quality clusters using:
  - Absolute coherence threshold: 0.45 (quality floor)
  - Relative coherence drop: 80% of median (flags underperforming clusters)
- **Smart labeling**: C-TF-IDF with (1,3)-grams for interpretable topic names
- **Optional AI enhancement**: Gemini API integration for improved topic labels (configurable via environment variable)
- **Monthly retraining**: Automated model updates to adapt to evolving discussions

### 💭 Sentiment Analysis
- Fine-tuned Indonesian BERT model: [`mdhugol/indonesia-bert-sentiment-classification`](https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification)
- Classifies opinions as positive, neutral, or negative

### 📍 Named Entity Recognition (NER)
- **Custom fine-tuned model**: [`Kiuyha/surabaya-opinion-indobert-ner`](https://huggingface.co/Kiuyha/surabaya-opinion-indobert-ner)
- **Base model**: [`indobenchmark/indobert-base-p2`](https://huggingface.co/indobenchmark/indobert-base-p2)
- **Training dataset**: Manually annotated by our team - [`Kiuyha/surabaya-ner-dataset`](https://huggingface.co/datasets/Kiuyha/surabaya-ner-dataset)
- Extracts location entities and urban issue mentions from Indonesian text

### 🎨 Interactive Dashboard
- **Streamlit-powered frontend** hosted on Hugging Face Spaces
- Real-time data visualization of sentiment trends, topic distributions, and location mentions
- Connects directly to Supabase for live data updates

## Project Structure

```
.
├── .github/workflows/          # CI/CD automation
│   ├── build_frontend.yml      # Frontend deployment
│   ├── monthly_retrain.yml     # Monthly topic model retraining
│   └── weekly_pipeline.yml     # Weekly data collection
├── frontend/                   # Streamlit dashboard
│   ├── app.py                  # Main application entry
│   ├── dashboard.py            # Visualization components
│   ├── dataview.py             # Data exploration interface
│   ├── diagnostics.py          # Model diagnostics view
│   └── requirements.txt
├── src/                        # Core pipeline modules
│   ├── config.py               # Configuration management
│   ├── core.py                 # Main pipeline orchestration
│   ├── geocoding.py            # Location processing (WIP)
│   ├── ner.py                  # NER extraction
│   ├── preprocess.py           # Text preprocessing
│   ├── scraper.py              # Data collection
│   ├── sentiment.py            # Sentiment analysis
│   ├── topics.py               # Topic modeling
│   ├── training/
│   │   └── train_topics_model.py  # Topic model training script
│   └── utils/
│       ├── gemini_api.py       # Gemini API integration
│       ├── types.py            # Type definitions
│       ├── update_preprocess.py
│       └── visualize.py        # Plotting utilities
├── notebook.ipynb              # Exploratory analysis
└── requirements.txt            # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Surabaya-Opinion-Analytics.git
cd Surabaya-Opinion-Analytics

# Install dependencies
pip install -r requirements.txt

# For frontend development
cd frontend
pip install -r requirements.txt
```

## Configuration

Set up your environment variables (create a `.env` file):

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
HF_TOKEN=your_huggingface_token

# optional if you want to label the topics using the Gemini API
USE_GEMINI_API=true 
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Run the Complete Pipeline

```bash
python -m src.main.py
```

This executes:
1. Data scraping from X and Reddit
2. Text preprocessing
3. Sentiment analysis
4. NER extraction
5. Topic modeling
6. Data export to Supabase

### Train Topic Model

```bash
python -m src.training.train_topics_model.py
```

### Launch Dashboard Locally

```bash
cd frontend
streamlit run app.py
```

## Topic Modeling Details

Our approach to topic discovery prioritizes both coherence and distinctiveness:

1. **Embedding Generation**: FastText creates dense vector representations of preprocessed text
2. **Clustering**: K-Means groups similar documents, testing multiple K values
3. **Scoring**: Each configuration receives a composite score balancing:
   - **Coherence (60%)**: How well words within a topic relate (C_v metric)
   - **Separation (40%)**: How distinct topics are from each other (cosine distance)
4. **Quality Control**: Clusters below coherence thresholds are automatically filtered out
5. **Labeling**: C-TF-IDF identifies characteristic n-grams (1-3 words) for each topic
6. **Optional Enhancement**: Gemini API can generate more natural topic descriptions

## Automated Workflows

- **Weekly Pipeline** (GitHub Actions): Scrapes new data, runs analysis, updates Supabase
- **Monthly Retraining**: Retrains topic model on accumulated data to capture emerging themes
- **Frontend Deployment**: Automatically deploys dashboard updates to Hugging Face Spaces

## Known Limitations

- Geocoding functionality is currently under development
- NER model is specifically tuned for Surabaya-related entities and may not generalize to other regions

## Models & Resources

- **Topic Model**: [Kiuyha/surabaya-opinion-tweet-clusters](https://huggingface.co/Kiuyha/surabaya-opinion-tweet-clusters)
- **NER Model**: [Kiuyha/surabaya-opinion-indobert-ner](https://huggingface.co/Kiuyha/surabaya-opinion-indobert-ner)
- **NER Dataset**: [Kiuyha/surabaya-ner-dataset](https://huggingface.co/datasets/Kiuyha/surabaya-ner-dataset)
- **Live Dashboard**: [Surabaya Opinion Analytics](https://huggingface.co/spaces/Kiuyha/Surabaya-Opinion-Analytics)

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- Manual NER annotation by our dedicated team
- Built with Hugging Face Transformers, FastText, Streamlit, and Supabase
- Sentiment model by mdhugol
- Base NER model from IndoBERT team

---

**Live Dashboard**: https://huggingface.co/spaces/Kiuyha/Surabaya-Opinion-Analytics