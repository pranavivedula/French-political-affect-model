# French Political Rhetoric NLP Classification System

An NLP system that analyzes French political rhetoric and maps major political parties onto the Circumplex Model of Affect (valence × arousal dimensions).

## Overview

This project scrapes official documents, press releases, and statements from 6 major French political parties, analyzes them using transformer-based NLP models, and visualizes party positions on a 2D affect space:

- **X-axis (Valence)**: Negative → Positive sentiment
- **Y-axis (Arousal)**: Low → High emotional activation

### Analyzed Parties

- **LREM** (La République En Marche / Renaissance) - Center
- **RN** (Rassemblement National) - Far-right
- **LFI** (La France Insoumise) - Far-left
- **PS** (Parti Socialiste) - Center-left
- **EELV** (Europe Écologie Les Verts) - Green/Center-left
- **LR** (Les Républicains) - Center-right

## Features

- Web scraping with robots.txt compliance and rate limiting
- French language NLP using multilingual transformers
- Valence and arousal detection
- Temporal tracking of party positions
- Interactive Streamlit dashboard with circumplex visualization
- Automated daily monitoring and updates

## Project Structure

```
french-political-affect/
├── config/                 # Configuration files
│   ├── settings.py        # Application settings
│   └── party_config.yaml  # Party metadata
├── data/                  # Data storage
│   ├── raw/              # Scraped HTML/text
│   ├── processed/        # Processed data
│   └── political_affect.db  # SQLite database
├── src/                   # Source code
│   ├── scraping/         # Web scraping modules
│   ├── nlp/              # NLP processing
│   ├── database/         # Database models and repositories
│   ├── analysis/         # Aggregation and analysis
│   └── pipeline/         # Orchestration and scheduling
├── dashboard/             # Streamlit dashboard
├── scripts/               # Utility scripts
└── notebooks/             # Jupyter notebooks for exploration
```

## Installation

### Prerequisites

- Python 3.9+
- pip and virtualenv

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd french-political-affect
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration if needed
   ```

5. **Initialize the database:**
   ```bash
   python scripts/setup_database.py
   ```

## Usage

### 1. Test Scraper

Test the scraper for a specific party:

```bash
python scripts/run_scraper.py --test --party LREM
```

### 2. Scrape Documents

Scrape documents from all parties:

```bash
python scripts/run_scraper.py --all --days 30
```

Scrape from a specific party:

```bash
python scripts/run_scraper.py --party LREM --days 7
```

### 3. Run NLP Analysis

Analyze scraped documents:

```bash
python scripts/run_analysis.py
```

### 4. Launch Dashboard

Start the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

### 5. Enable Automated Monitoring

To enable daily automated scraping and analysis, set in your `.env`:

```
ENABLE_SCHEDULER=true
```

Then run the main application with scheduling enabled.

## Database Schema

### Tables

- **parties**: Political party metadata
- **documents**: Scraped political documents
- **sentences**: Individual sentences with scores
- **temporal_snapshots**: Historical party positions over time
- **scraping_logs**: Logs of scraping operations

## NLP Pipeline

1. **Text Preprocessing**: Clean, tokenize, segment sentences
2. **Valence Detection**: Predict sentiment (-1 to +1)
3. **Arousal Detection**: Predict emotional activation (0 to 1)
4. **Aggregation**: Combine scores at sentence → document → party levels
5. **Temporal Tracking**: Create monthly/weekly snapshots

### Models

- **Default**: Pretrained multilingual transformers (XLM-RoBERTa)
- **Valence**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Arousal**: Custom fine-tuned model (future enhancement)

## Dashboard Features

- **Circumplex Plot**: 2D scatter plot showing party positions
- **Temporal Evolution**: Line charts tracking changes over time
- **Party Details**: Statistics and sample documents for each party
- **Filters**: Date range, aggregation method selection
- **Export**: Download results as CSV/JSON

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ dashboard/ scripts/
flake8 src/ dashboard/ scripts/
```

### Adding a New Party

1. Add party metadata to `config/party_config.yaml`
2. Create party-specific scraper in `src/scraping/party_scrapers.py`
3. Run database setup to add the party: `python scripts/setup_database.py`

## Methodology

### Circumplex Model of Affect

Based on Russell's (1980) Circumplex Model, which represents emotions in a 2D space:

- **Valence** (X-axis): Pleasantness/unpleasantness of the affect
- **Arousal** (Y-axis): Degree of activation or physiological arousal

### Aggregation Strategy

Party-level positions are computed using weighted averages:

```
weight = recency_factor (0.7) + length_factor (0.3)
```

- **Recency**: Exponential decay with 60-day half-life
- **Length**: Normalized by word count (capped at 1000 words)

### Validation

- Manual annotation of sample texts
- Comparison with political science literature
- Face validity checks (expected patterns)

## Ethical Considerations

- **GDPR Compliance**: Only public party statements (no personal data)
- **robots.txt**: Full compliance with crawling directives
- **Rate Limiting**: 3-5 seconds between requests
- **Transparency**: Methodology fully documented

## Future Enhancements

- [ ] Fine-tune models on French political texts
- [ ] Topic-specific analysis (immigration, economy, etc.)
- [ ] Social media integration (Twitter/X)
- [ ] Multi-country comparison
- [ ] Event detection and correlation

## License

MIT License - See LICENSE file for details

## Citation

If you use this system in research, please cite:

```
French Political Rhetoric NLP Classification System
https://github.com/yourusername/french-political-affect
```

## Contact

For questions or issues, please open an issue on GitHub.

---

**Disclaimer**: This tool is for research and educational purposes. Political positions are computationally derived and may not reflect nuanced political realities.
