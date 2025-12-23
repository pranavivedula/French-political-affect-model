"""
French Political Affect Analysis Dashboard

Main Streamlit application for visualizing party positions on the Circumplex Model of Affect.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from datetime import datetime, timedelta
from loguru import logger

# Import dashboard components
from dashboard.components.circumplex_plot import create_circumplex_plot, create_comparison_plot
from dashboard.components.timeline import create_timeline_plot, create_trajectory_plot
from dashboard.components.party_details import (
    display_party_card,
    display_party_comparison,
    display_recent_documents
)
from dashboard.utils.styling import (
    apply_custom_css,
    create_methodology_section,
    create_about_section,
    create_quadrant_guide
)

# Import data access
from src.database.repository import (
    DatabaseRepository,
    PartyRepository,
    DocumentRepository,
    TemporalSnapshotRepository
)
from scripts.compute_party_positions import compute_all_parties, compute_party_position
from config.settings import load_party_config


# Page configuration
st.set_page_config(
    page_title="French Political Affect Analysis",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_css()


@st.cache_data(ttl=3600)
def load_party_positions(start_date=None, end_date=None, method='weighted'):
    """Load party positions with caching."""
    try:
        results = compute_all_parties(
            start_date=start_date,
            end_date=end_date,
            method=method
        )
        return results
    except Exception as e:
        logger.error(f"Error loading party positions: {e}")
        return []


@st.cache_data(ttl=3600)
def load_temporal_data(party_codes=None):
    """Load temporal snapshots with caching."""
    db = DatabaseRepository()
    snapshot_repo = TemporalSnapshotRepository(db)
    party_repo = PartyRepository(db)

    temporal_data = {}

    # Get all parties if none specified
    if not party_codes:
        party_codes = [p['code'] for p in load_party_config()]

    for party_code in party_codes:
        party = party_repo.get_party_by_code(party_code)
        if not party:
            continue

        snapshots = snapshot_repo.get_snapshots_by_party(party.id)

        if snapshots:
            temporal_data[party_code] = [
                {
                    'snapshot_date': s.snapshot_date,
                    'valence': s.valence,
                    'arousal': s.arousal,
                    'party_name': party.name,
                    'color': party.color,
                    'period_start': s.period_start,
                    'period_end': s.period_end,
                    'num_documents': s.num_documents
                }
                for s in snapshots
            ]

    return temporal_data


def main():
    """Main dashboard application."""

    # Header
    st.title("🗳️ French Political Affect Analysis")
    st.markdown("### Mapping Political Rhetoric on the Circumplex Model of Affect")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Date range filter
        st.subheader("Date Range")
        use_date_filter = st.checkbox("Filter by date range", value=False)

        start_date = None
        end_date = None

        if use_date_filter:
            days_back = st.slider(
                "Days to look back",
                min_value=7,
                max_value=365,
                value=90,
                step=7
            )
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            st.info(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Aggregation method
        st.subheader("Aggregation Method")
        method = st.selectbox(
            "Select method",
            options=['weighted', 'mean', 'median'],
            index=0,
            help="Weighted: combines recency and document length"
        )

        # Display options
        st.subheader("Display Options")
        show_ci = st.checkbox("Show confidence intervals", value=True)
        show_quadrant_guide = st.checkbox("Show quadrant guide", value=False)

        # Refresh data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()

        # About sections
        st.markdown("---")
        create_methodology_section()
        create_about_section()

    # Load data
    with st.spinner("Loading party positions..."):
        party_positions = load_party_positions(
            start_date=start_date,
            end_date=end_date,
            method=method
        )

    if not party_positions:
        st.error("❌ No data available. Please run the analysis pipeline first:")
        st.code("""
# 1. Scrape documents
python scripts/run_scraper.py --all --days 90

# 2. Analyze documents
python scripts/run_analysis.py

# 3. Refresh this dashboard
        """)
        return

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Circumplex Model",
        "📈 Temporal Evolution",
        "🔍 Party Details",
        "📝 Documents"
    ])

    # Tab 1: Circumplex Plot
    with tab1:
        st.header("Party Positions on Circumplex Model")

        if show_quadrant_guide:
            create_quadrant_guide()

        # Main circumplex plot
        fig = create_circumplex_plot(party_positions, show_ci=show_ci)
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_valence = sum(p['valence'] for p in party_positions) / len(party_positions)
            st.metric("Average Valence", f"{avg_valence:.3f}")

        with col2:
            avg_arousal = sum(p['arousal'] for p in party_positions) / len(party_positions)
            st.metric("Average Arousal", f"{avg_arousal:.3f}")

        with col3:
            total_docs = sum(p['num_documents'] for p in party_positions)
            st.metric("Total Documents", f"{total_docs:,}")

        # Comparison plots
        st.subheader("Dimensional Comparisons")
        col1, col2 = st.columns(2)

        with col1:
            fig_valence = create_comparison_plot(party_positions, dimension='valence')
            st.plotly_chart(fig_valence, use_container_width=True)

        with col2:
            fig_arousal = create_comparison_plot(party_positions, dimension='arousal')
            st.plotly_chart(fig_arousal, use_container_width=True)

    # Tab 2: Temporal Evolution
    with tab2:
        st.header("Temporal Evolution of Party Affect")

        # Load temporal data
        with st.spinner("Loading temporal data..."):
            temporal_data = load_temporal_data()

        if temporal_data and any(temporal_data.values()):
            # Timeline plot
            st.subheader("Valence and Arousal Over Time")
            fig_timeline = create_timeline_plot(temporal_data)
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Trajectory plot
            st.subheader("Movement in Circumplex Space")
            fig_trajectory = create_trajectory_plot(temporal_data)
            st.plotly_chart(fig_trajectory, use_container_width=True)

        else:
            st.warning("⚠️ No temporal data available. Generate snapshots using:")
            st.code("python scripts/compute_party_positions.py --all --save-snapshots")

    # Tab 3: Party Details
    with tab3:
        st.header("Detailed Party Analysis")

        # Party selector
        party_codes = [p['party_code'] for p in party_positions]
        selected_party_code = st.selectbox(
            "Select party to view details",
            options=party_codes,
            format_func=lambda x: next(
                (p['party_name'] for p in party_positions if p['party_code'] == x),
                x
            )
        )

        # Find selected party data
        selected_party = next(
            (p for p in party_positions if p['party_code'] == selected_party_code),
            None
        )

        if selected_party:
            display_party_card(selected_party)

            # Comparison with another party
            st.markdown("---")
            st.subheader("Compare with Another Party")

            compare_party_code = st.selectbox(
                "Select party to compare",
                options=[code for code in party_codes if code != selected_party_code],
                format_func=lambda x: next(
                    (p['party_name'] for p in party_positions if p['party_code'] == x),
                    x
                )
            )

            compare_party = next(
                (p for p in party_positions if p['party_code'] == compare_party_code),
                None
            )

            if compare_party:
                display_party_comparison(selected_party, compare_party)

    # Tab 4: Documents
    with tab4:
        st.header("Recent Analyzed Documents")

        # Load recent documents
        db = DatabaseRepository()
        doc_repo = DocumentRepository(db)

        # Filters
        col1, col2 = st.columns(2)

        with col1:
            filter_party = st.selectbox(
                "Filter by party",
                options=['All'] + party_codes,
                index=0
            )

        with col2:
            num_docs = st.slider(
                "Number of documents",
                min_value=10,
                max_value=100,
                value=20,
                step=10
            )

        # Fetch documents
        if filter_party == 'All':
            recent_docs = doc_repo.get_recent_documents(limit=num_docs)
        else:
            party = next((p for p in party_positions if p['party_code'] == filter_party), None)
            if party:
                party_repo = PartyRepository(db)
                party_obj = party_repo.get_party_by_code(filter_party)
                if party_obj:
                    recent_docs = doc_repo.get_documents_by_party(
                        party_obj.id,
                        analyzed_only=True
                    )[:num_docs]
                else:
                    recent_docs = []
            else:
                recent_docs = []

        # Convert to dicts for display
        docs_data = [
            {
                'title': doc.title,
                'party_code': doc.party.code if doc.party else 'N/A',
                'date_published': doc.date_published.strftime('%Y-%m-%d') if doc.date_published else 'N/A',
                'valence': doc.valence,
                'arousal': doc.arousal,
                'word_count': doc.word_count,
                'url': doc.url,
                'content': doc.content
            }
            for doc in recent_docs
        ]

        display_recent_documents(docs_data, limit=num_docs)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>French Political Affect Analysis System</p>
            <p>🤖 Generated with <a href='https://claude.com/claude-code'>Claude Code</a></p>
            <p><small>For research and educational purposes only</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
