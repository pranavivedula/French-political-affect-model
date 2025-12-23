"""Timeline visualization components for temporal evolution."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import pandas as pd


def create_timeline_plot(temporal_data: Dict[str, List[Dict]]) -> go.Figure:
    """
    Create a timeline plot showing party evolution over time.

    Args:
        temporal_data: Dictionary mapping party codes to list of temporal snapshots

    Returns:
        Plotly figure with subplots for valence and arousal
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Valence Over Time', 'Arousal Over Time'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    # Track if any data was added
    has_data = False

    for party_code, snapshots in temporal_data.items():
        if not snapshots:
            continue

        has_data = True

        # Extract data
        dates = [s['snapshot_date'] for s in snapshots]
        valences = [s['valence'] for s in snapshots]
        arousals = [s['arousal'] for s in snapshots]
        party_name = snapshots[0].get('party_name', party_code)
        color = snapshots[0].get('color', '#888888')

        # Valence line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=valences,
                mode='lines+markers',
                name=party_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             'Valence: %{y:.3f}<br>' +
                             '<extra></extra>',
                showlegend=True,
                legendgroup=party_code
            ),
            row=1, col=1
        )

        # Arousal line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=arousals,
                mode='lines+markers',
                name=party_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             'Arousal: %{y:.3f}<br>' +
                             '<extra></extra>',
                showlegend=False,
                legendgroup=party_code
            ),
            row=2, col=1
        )

    # Update axes
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(
        title_text='Valence',
        row=1, col=1,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='lightgray',
        range=[-1.2, 1.2]
    )
    fig.update_yaxes(
        title_text='Arousal',
        row=2, col=1,
        range=[0, 1]
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Temporal Evolution of Party Affect</b>',
            x=0.5,
            xanchor='center'
        ),
        height=700,
        hovermode='x unified',
        plot_bgcolor='white',
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

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    if not has_data:
        fig.add_annotation(
            text='No temporal data available',
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray')
        )

    return fig


def create_trajectory_plot(temporal_data: Dict[str, List[Dict]]) -> go.Figure:
    """
    Create a 2D trajectory plot showing movement in circumplex space over time.

    Args:
        temporal_data: Dictionary mapping party codes to list of temporal snapshots

    Returns:
        Plotly figure showing trajectories
    """
    fig = go.Figure()

    # Add reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = (np.sin(theta) + 1) / 2

    fig.add_trace(go.Scatter(
        x=circle_x,
        y=circle_y,
        mode='lines',
        line=dict(color='lightgray', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Plot trajectories for each party
    for party_code, snapshots in temporal_data.items():
        if not snapshots or len(snapshots) < 2:
            continue

        valences = [s['valence'] for s in snapshots]
        arousals = [s['arousal'] for s in snapshots]
        dates = [s['snapshot_date'].strftime('%Y-%m') for s in snapshots]
        party_name = snapshots[0].get('party_name', party_code)
        color = snapshots[0].get('color', '#888888')

        # Trajectory line
        fig.add_trace(go.Scatter(
            x=valences,
            y=arousals,
            mode='lines+markers',
            name=party_name,
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol='circle'),
            text=dates,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{text}<br>' +
                         'Valence: %{x:.3f}<br>' +
                         'Arousal: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))

        # Add arrow to show direction (from first to last point)
        if len(valences) >= 2:
            fig.add_annotation(
                x=valences[-1],
                y=arousals[-1],
                ax=valences[-2],
                ay=arousals[-2],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Party Trajectories in Circumplex Space</b>',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Valence (Negative ← → Positive)',
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinewidth=2,
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            title='Arousal (Low ← → High)',
            range=[0, 1],
            gridcolor='#f0f0f0'
        ),
        height=600,
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02
        ),
        margin=dict(l=80, r=200, t=80, b=80)
    )

    return fig


import numpy as np  # Add this at the top if not already there
