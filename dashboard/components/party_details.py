"""Party details display components."""

import streamlit as st
from typing import Dict, List


def display_party_card(party_data: Dict):
    """
    Display a detailed card for a party.

    Args:
        party_data: Dictionary with party information
    """
    party_name = party_data.get('party_name', 'Unknown')
    party_code = party_data.get('party_code', '')
    color = party_data.get('color', '#888888')

    # Create colored header
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h2 style="color: white; margin: 0;">{party_name} ({party_code})</h2>
            <p style="color: white; margin: 5px 0 0 0; font-size: 14px;">
                {party_data.get('political_position', '').title()}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        valence = party_data.get('valence', 0)
        valence_ci = party_data.get('valence_ci', 0)
        st.metric(
            "Valence",
            f"{valence:.3f}",
            delta=None,
            help=f"95% CI: ±{valence_ci:.3f}"
        )

    with col2:
        arousal = party_data.get('arousal', 0)
        arousal_ci = party_data.get('arousal_ci', 0)
        st.metric(
            "Arousal",
            f"{arousal:.3f}",
            delta=None,
            help=f"95% CI: ±{arousal_ci:.3f}"
        )

    with col3:
        num_docs = party_data.get('num_documents', 0)
        st.metric(
            "Documents",
            f"{num_docs:,}",
            delta=None
        )

    # Circumplex information
    circumplex = party_data.get('circumplex', {})

    st.markdown("### Circumplex Position")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Quadrant:** {circumplex.get('quadrant_label', 'N/A')}")
        st.markdown(f"**Nearest Emotion:** {circumplex.get('nearest_emotion', 'N/A').title()}")

    with col2:
        st.markdown(f"**Angle:** {circumplex.get('angle', 0):.1f}°")
        st.markdown(f"**Magnitude:** {circumplex.get('magnitude', 0):.3f}")

    # Statistical details
    if st.expander("📊 Statistical Details", expanded=False):
        valence_stats = party_data.get('valence_stats', {})
        arousal_stats = party_data.get('arousal_stats', {})

        st.markdown("**Valence Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Mean: {valence_stats.get('mean', 0):.3f}")
            st.write(f"Std: {valence_stats.get('std', 0):.3f}")
        with col2:
            st.write(f"Min: {valence_stats.get('min', 0):.3f}")
            st.write(f"Max: {valence_stats.get('max', 0):.3f}")
        with col3:
            st.write(f"Q25: {valence_stats.get('q25', 0):.3f}")
            st.write(f"Q75: {valence_stats.get('q75', 0):.3f}")

        st.markdown("**Arousal Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Mean: {arousal_stats.get('mean', 0):.3f}")
            st.write(f"Std: {arousal_stats.get('std', 0):.3f}")
        with col2:
            st.write(f"Min: {arousal_stats.get('min', 0):.3f}")
            st.write(f"Max: {arousal_stats.get('max', 0):.3f}")
        with col3:
            st.write(f"Q25: {arousal_stats.get('q25', 0):.3f}")
            st.write(f"Q75: {arousal_stats.get('q75', 0):.3f}")


def display_party_comparison(party1_data: Dict, party2_data: Dict):
    """
    Display a comparison between two parties.

    Args:
        party1_data: First party data
        party2_data: Second party data
    """
    st.markdown("### Party Comparison")

    # Names
    party1_name = party1_data.get('party_name', 'Party 1')
    party2_name = party2_data.get('party_name', 'Party 2')

    # Calculate differences
    valence_diff = party2_data.get('valence', 0) - party1_data.get('valence', 0)
    arousal_diff = party2_data.get('arousal', 0) - party1_data.get('arousal', 0)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{party1_name}**")
        st.write(f"Valence: {party1_data.get('valence', 0):.3f}")
        st.write(f"Arousal: {party1_data.get('arousal', 0):.3f}")

    with col2:
        st.markdown(f"**{party2_name}**")
        st.write(f"Valence: {party2_data.get('valence', 0):.3f}")
        st.write(f"Arousal: {party2_data.get('arousal', 0):.3f}")

    st.markdown("---")
    st.markdown("**Differences:**")

    if abs(valence_diff) > 0.1:
        more_positive = party2_name if valence_diff > 0 else party1_name
        st.write(f"• {more_positive} is more positive (Δ = {abs(valence_diff):.3f})")
    else:
        st.write(f"• Similar valence (Δ = {abs(valence_diff):.3f})")

    if abs(arousal_diff) > 0.1:
        more_aroused = party2_name if arousal_diff > 0 else party1_name
        st.write(f"• {more_aroused} shows higher arousal (Δ = {abs(arousal_diff):.3f})")
    else:
        st.write(f"• Similar arousal (Δ = {abs(arousal_diff):.3f})")


def display_recent_documents(documents: List[Dict], limit: int = 10):
    """
    Display a table of recent analyzed documents.

    Args:
        documents: List of document dictionaries
        limit: Maximum number to display
    """
    st.markdown("### Recent Analyzed Documents")

    if not documents:
        st.info("No documents to display")
        return

    # Sort by date (most recent first)
    sorted_docs = sorted(
        documents,
        key=lambda d: d.get('date_published', ''),
        reverse=True
    )[:limit]

    # Display as table
    for i, doc in enumerate(sorted_docs, 1):
        with st.expander(f"{i}. {doc.get('title', 'Untitled')[:80]}..."):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Party:** {doc.get('party_code', 'N/A')}")
                st.write(f"**Date:** {doc.get('date_published', 'N/A')}")

            with col2:
                st.write(f"**Valence:** {doc.get('valence', 0):.3f}")
                st.write(f"**Arousal:** {doc.get('arousal', 0):.3f}")

            with col3:
                st.write(f"**Words:** {doc.get('word_count', 0):,}")
                if doc.get('url'):
                    st.markdown(f"[View Source]({doc.get('url')})")

            # Preview
            content_preview = doc.get('content', '')[:300] + '...'
            st.markdown(f"**Preview:** {content_preview}")


def display_extreme_documents(documents: List[Dict], metric: str = 'valence', top_n: int = 5):
    """
    Display documents with extreme scores.

    Args:
        documents: List of documents
        metric: 'valence' or 'arousal'
        top_n: Number of extremes to show from each end
    """
    if not documents:
        st.info("No documents available")
        return

    # Sort by metric
    sorted_docs = sorted(documents, key=lambda d: d.get(metric, 0))

    # Get extremes
    lowest = sorted_docs[:top_n]
    highest = sorted_docs[-top_n:][::-1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Lowest {metric.title()}")
        for i, doc in enumerate(lowest, 1):
            st.markdown(f"**{i}. {doc.get('title', 'Untitled')[:40]}...**")
            st.write(f"{metric.title()}: {doc.get(metric, 0):.3f}")
            st.write(f"Party: {doc.get('party_code', 'N/A')}")
            st.markdown("---")

    with col2:
        st.markdown(f"### Highest {metric.title()}")
        for i, doc in enumerate(highest, 1):
            st.markdown(f"**{i}. {doc.get('title', 'Untitled')[:40]}...**")
            st.write(f"{metric.title()}: {doc.get(metric, 0):.3f}")
            st.write(f"Party: {doc.get('party_code', 'N/A')}")
            st.markdown("---")
