"""Dashboard styling and utility functions."""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit dashboard."""
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding-top: 2rem;
        }

        /* Headers */
        h1 {
            color: #1f77b4;
            font-family: 'Arial', sans-serif;
        }

        h2 {
            color: #2c3e50;
            margin-top: 1.5rem;
        }

        h3 {
            color: #34495e;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: #f8f9fa;
        }

        /* Info boxes */
        .stAlert {
            border-radius: 10px;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 20px;
            padding: 0.5rem 2rem;
            font-weight: 600;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Tables */
        .dataframe {
            font-size: 0.9rem;
        }

        /* Custom party cards */
        .party-card {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)


def create_info_box(title: str, content: str, icon: str = "ℹ️"):
    """
    Create a styled information box.

    Args:
        title: Box title
        content: Box content
        icon: Icon to display
    """
    st.markdown(f"""
        <div style="
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        ">
            <h4 style="margin: 0; color: #1976d2;">{icon} {title}</h4>
            <p style="margin: 5px 0 0 0; color: #424242;">{content}</p>
        </div>
    """, unsafe_allow_html=True)


def create_methodology_section():
    """Create an expandable methodology section."""
    with st.expander("📖 Methodology", expanded=False):
        st.markdown("""
        ### About the Circumplex Model of Affect

        This dashboard visualizes French political rhetoric using Russell's (1980) **Circumplex Model of Affect**,
        which represents emotions in a two-dimensional space:

        - **Valence (X-axis):** Pleasantness or unpleasantness of affect
          - -1.0 = Very negative (criticism, anger, pessimism)
          - 0.0 = Neutral
          - +1.0 = Very positive (praise, optimism, enthusiasm)

        - **Arousal (Y-axis):** Degree of activation or physiological arousal
          - 0.0 = Low activation (calm, passive, low energy)
          - 0.5 = Moderate activation
          - 1.0 = High activation (excited, urgent, intense)

        ### Data Processing Pipeline

        1. **Data Collection:** Web scraping of official party websites (press releases, statements, manifestos)
        2. **NLP Analysis:** Multilingual transformer models analyze each document for valence and arousal
        3. **Aggregation:** Document-level scores are weighted by recency and length, then aggregated to party level
        4. **Visualization:** Parties are plotted on the circumplex model with confidence intervals

        ### Aggregation Method

        Party positions are computed using weighted averages:
        - **Recency weight (70%):** Exponential decay with 60-day half-life
        - **Length weight (30%):** Normalized by word count (capped at 1000 words)

        ### Statistical Confidence

        - **95% Confidence Intervals:** Error bars show statistical uncertainty
        - **Minimum documents:** At least 10 documents required for reliable aggregation
        """)


def create_about_section():
    """Create an about section."""
    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown("""
        ### French Political Affect Analysis System

        This system analyzes rhetoric from 6 major French political parties:
        - **LREM** (La République En Marche / Renaissance) - Center
        - **RN** (Rassemblement National) - Far-right
        - **LFI** (La France Insoumise) - Far-left
        - **PS** (Parti Socialiste) - Center-left
        - **EELV** (Europe Écologie Les Verts) - Green/Center-left
        - **LR** (Les Républicains) - Center-right

        ### Technology Stack

        - **NLP Models:** XLM-RoBERTa multilingual transformers
        - **Database:** SQLite/PostgreSQL
        - **Dashboard:** Streamlit + Plotly
        - **Analysis:** NumPy, Pandas, SciPy

        ### Ethical Considerations

        - Only public party statements analyzed (no personal data)
        - Full compliance with robots.txt and GDPR
        - Rate limiting and respectful scraping practices
        - Methodology fully documented and transparent

        ### Disclaimer

        This tool is for research and educational purposes. Political positions are computationally
        derived and may not reflect nuanced political realities. Results should be interpreted with
        appropriate context and domain knowledge.

        ---

        **Generated with Claude Code**
        https://github.com/pranavivedula/French-political-affect-model
        """)


def format_date_range(start_date, end_date):
    """
    Format a date range for display.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        Formatted string
    """
    if start_date and end_date:
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    return "All time"


def get_emotion_emoji(emotion: str) -> str:
    """
    Get emoji for an emotion.

    Args:
        emotion: Emotion name

    Returns:
        Emoji string
    """
    emoji_map = {
        'excited': '🤩',
        'alert': '👀',
        'elated': '😄',
        'happy': '😊',
        'content': '😌',
        'calm': '😇',
        'relaxed': '😎',
        'serene': '🧘',
        'bored': '😐',
        'depressed': '😔',
        'sad': '😢',
        'upset': '😠',
        'stressed': '😰',
        'tense': '😬',
        'nervous': '😨',
        'angry': '😡'
    }
    return emoji_map.get(emotion.lower(), '💭')


def create_quadrant_guide():
    """Create a visual guide to the quadrants."""
    st.markdown("""
    ### Circumplex Quadrants

    | Quadrant | Valence | Arousal | Characteristics | Examples |
    |----------|---------|---------|-----------------|----------|
    | **Q1** | Positive | High | Energetic, optimistic, mobilizing | Excited, Alert, Elated |
    | **Q2** | Negative | High | Urgent criticism, combative | Angry, Stressed, Tense |
    | **Q3** | Negative | Low | Resigned, pessimistic, passive | Sad, Depressed, Bored |
    | **Q4** | Positive | Low | Calm confidence, satisfaction | Content, Relaxed, Serene |
    """)
