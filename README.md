# ResearchNavigator v2

ResearchNavigator is a Streamlit-based research paper search and analysis platform. It integrates paper search, Korean/English translation, summarization, bookmarks, and chemical engineering research analysis.

## Main Features

- PubMed / arXiv / Semantic Scholar / Google Scholar integrated search
- Korean query translation and Korean summary support
- Bookmark management and export
- School Wi-Fi friendly retry/session network layer
- Chemical engineering tag analysis
- Yearly research trend visualization
- Keyword frequency and keyword co-occurrence network
- TF-IDF based similar paper recommendation
- PDF upload text extraction, keyword extraction, and simple summarization

## Chemical Engineering Focus

The v2 dashboard is designed for chemical engineering topics such as batteries, catalysts, hydrogen, CCUS, membrane separation, polymers, semiconductor processes, environmental engineering, and process optimization.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

Google Scholar and Google Translate may fail on some school networks. The app records network errors in `network_errors.log` and continues with available sources.

Do not upload real user data in `users.json` or `bookmarks.json` when publishing publicly.
