"""Circumplex model visualization component."""

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict


def create_circumplex_plot(party_positions: List[Dict], show_ci: bool = True) -> go.Figure:
    """
    Create an interactive circumplex model plot with party positions.

    Args:
        party_positions: List of dictionaries with party data
        show_ci: Whether to show confidence interval error bars

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add reference circle for circumplex
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = (np.sin(theta) + 1) / 2  # Scale to 0-1 for arousal

    fig.add_trace(go.Scatter(
        x=circle_x,
        y=circle_y,
        mode='lines',
        line=dict(color='lightgray', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
        name='Reference Circle'
    ))

    # Add quadrant labels
    annotations = [
        dict(
            x=0.7, y=0.85,
            text="<b>High Activation<br>Positive Valence</b><br><i>Excited, Alert</i>",
            showarrow=False,
            font=dict(size=10, color='gray'),
            align='center'
        ),
        dict(
            x=-0.7, y=0.85,
            text="<b>High Activation<br>Negative Valence</b><br><i>Angry, Stressed</i>",
            showarrow=False,
            font=dict(size=10, color='gray'),
            align='center'
        ),
        dict(
            x=-0.7, y=0.15,
            text="<b>Low Activation<br>Negative Valence</b><br><i>Sad, Depressed</i>",
            showarrow=False,
            font=dict(size=10, color='gray'),
            align='center'
        ),
        dict(
            x=0.7, y=0.15,
            text="<b>Low Activation<br>Positive Valence</b><br><i>Calm, Content</i>",
            showarrow=False,
            font=dict(size=10, color='gray'),
            align='center'
        ),
    ]

    # Add center crosshairs
    fig.add_trace(go.Scatter(
        x=[-1, 1],
        y=[0.5, 0.5],
        mode='lines',
        line=dict(color='lightgray', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, 1],
        mode='lines',
        line=dict(color='lightgray', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Plot parties
    for party in party_positions:
        valence = party.get('valence', 0)
        arousal = party.get('arousal', 0.5)
        party_name = party.get('party_name', party.get('party_code', 'Unknown'))
        party_code = party.get('party_code', '')
        color = party.get('color', '#888888')
        num_docs = party.get('num_documents', 0)

        # Hover text
        hover_text = (
            f"<b>{party_name}</b><br>"
            f"Valence: {valence:.3f}<br>"
            f"Arousal: {arousal:.3f}<br>"
            f"Documents: {num_docs}<br>"
            f"Emotion: {party.get('circumplex', {}).get('nearest_emotion', 'N/A')}"
        )

        # Error bars for confidence intervals
        error_x = None
        error_y = None
        if show_ci:
            valence_ci = party.get('valence_ci', 0)
            arousal_ci = party.get('arousal_ci', 0)
            if valence_ci > 0:
                error_x = dict(type='data', array=[valence_ci], color=color, thickness=1.5)
            if arousal_ci > 0:
                error_y = dict(type='data', array=[arousal_ci], color=color, thickness=1.5)

        # Add party point
        fig.add_trace(go.Scatter(
            x=[valence],
            y=[arousal],
            mode='markers+text',
            marker=dict(
                size=20,
                color=color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=party_code,
            textposition='top center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            name=party_name,
            hovertext=hover_text,
            hoverinfo='text',
            error_x=error_x,
            error_y=error_y
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>French Political Parties: Circumplex Model of Affect</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis=dict(
            title='<b>Valence</b><br>(Negative ← → Positive)',
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgray',
            gridcolor='#f0f0f0',
            showgrid=True,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0<br>Very Negative', '-0.5', '0<br>Neutral', '0.5', '1.0<br>Very Positive']
        ),
        yaxis=dict(
            title='<b>Arousal</b><br>(Low Activation ← → High Activation)',
            range=[0, 1],
            gridcolor='#f0f0f0',
            showgrid=True,
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=['0<br>Very Low', '0.25', '0.5<br>Moderate', '0.75', '1.0<br>Very High']
        ),
        height=700,
        hovermode='closest',
        plot_bgcolor='white',
        annotations=annotations,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        margin=dict(l=80, r=200, t=80, b=80)
    )

    return fig


def create_comparison_plot(party_positions: List[Dict], dimension: str = 'valence') -> go.Figure:
    """
    Create a bar chart comparing parties on a single dimension.

    Args:
        party_positions: List of party position dictionaries
        dimension: 'valence' or 'arousal'

    Returns:
        Plotly figure object
    """
    # Sort parties by the dimension
    sorted_parties = sorted(
        party_positions,
        key=lambda p: p.get(dimension, 0),
        reverse=True
    )

    party_names = [p.get('party_code', '') for p in sorted_parties]
    values = [p.get(dimension, 0) for p in sorted_parties]
    colors = [p.get('color', '#888888') for p in sorted_parties]
    errors = [p.get(f'{dimension}_ci', 0) for p in sorted_parties]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=party_names,
        y=values,
        marker_color=colors,
        error_y=dict(
            type='data',
            array=errors,
            visible=True
        ),
        hovertemplate='<b>%{x}</b><br>' +
                     f'{dimension.capitalize()}: %{{y:.3f}}<br>' +
                     '<extra></extra>'
    ))

    title = 'Valence Comparison' if dimension == 'valence' else 'Arousal Comparison'
    y_title = 'Valence (Negative to Positive)' if dimension == 'valence' else 'Arousal (Low to High)'

    fig.update_layout(
        title=dict(
            text=f'<b>{title} Across Parties</b>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Party',
        yaxis_title=y_title,
        height=400,
        plot_bgcolor='white',
        showlegend=False,
        yaxis=dict(
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )
    )

    return fig
