"""
CX-Intel Dashboard - Customer Experience Intelligence Platform
A production-ready SaaS analytics dashboard for customer feedback analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from enum import Enum

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="CX-Intel Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SaaS-style dark theme
st.markdown("""
<style>
    :root {
        --primary-color: #3B82F6;
        --secondary-color: #8B5CF6;
        --success-color: #10B981;
        --danger-color: #EF4444;
        --warning-color: #F59E0B;
        --neutral-color: #6B7280;
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --bg-tertiary: #334155;
        --text-primary: #F1F5F9;
        --text-secondary: #CBD5E1;
        --border-color: #475569;
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }

    body {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    .main {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }

    .stMetricValue {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    .stMetricLabel {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* Card styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        border-color: var(--primary-color);
        box-shadow: 0 8px 12px rgba(59, 130, 246, 0.15);
    }

    .custom-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .trend-up {
        color: var(--success-color);
        font-weight: 600;
    }

    .trend-down {
        color: var(--danger-color);
        font-weight: 600;
    }

    .badge-high {
        display: inline-block;
        background-color: var(--danger-color);
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-medium {
        display: inline-block;
        background-color: var(--warning-color);
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-low {
        display: inline-block;
        background-color: var(--success-color);
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .sidebar-section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 3px solid var(--primary-color);
    }

    .sidebar-item {
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: var(--text-secondary);
        font-size: 0.95rem;
    }

    .sidebar-item:hover {
        background-color: var(--bg-tertiary);
        color: var(--primary-color);
    }

    .sidebar-item.active {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: var(--text-primary);
    }

    h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
    }

    h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    /* Alert styling */
    .alert-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-left: 4px solid var(--danger-color);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: var(--text-primary);
    }

    .insight-text {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }

    /* Plotly styling */
    .plotly {
        background-color: transparent !important;
    }

    div[data-testid="stPlotlyChart"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION & CACHING
# ============================================================================

@st.cache_data
def generate_dashboard_data():
    """Generate realistic dashboard data"""
    np.random.seed(42)
    
    # Time series data (30 days)
    days = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    sentiment_data = {
        'date': days,
        'positive': np.random.randint(800, 1200, 30),
        'neutral': np.random.randint(1500, 2500, 30),
        'negative': np.random.randint(600, 1000, 30),
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Current totals
    total_feedback = 10247
    positive_count = 3127
    neutral_count = 4892
    negative_count = 2228
    
    # Issue clusters data
    issues_data = {
        'Issue': [
            'Wait Time / Delays',
            'Staff Communication',
            'Billing / Charges',
            'Service Quality',
            'Cleanliness',
            'Doctor Availability',
            'Equipment Issues'
        ],
        'Count': [1850, 1420, 980, 756, 642, 538, 421],
        'Sentiment_Negative': [1650, 980, 720, 540, 380, 290, 210]
    }
    issues_df = pd.DataFrame(issues_data)
    
    return {
        'sentiment_df': sentiment_df,
        'issues_df': issues_df,
        'total_feedback': total_feedback,
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
    }

@st.cache_data
def generate_cascade_data():
    """Generate cause-effect cascade data"""
    return {
        'chains': [
            {
                'title': 'Primary Issue Chain',
                'steps': [
                    {'label': 'Long Wait Times', 'color': '#EF4444', 'impact': 'Direct'},
                    {'label': 'Inadequate Staffing', 'color': '#F59E0B', 'impact': 'Root'},
                    {'label': 'Scheduling Inefficiencies', 'color': '#F59E0B', 'impact': 'Process'},
                    {'label': 'Customer Frustration', 'color': '#EF4444', 'impact': 'Outcome'},
                ]
            },
            {
                'title': 'Communication Issue Chain',
                'steps': [
                    {'label': 'Poor Communication', 'color': '#EF4444', 'impact': 'Direct'},
                    {'label': 'Unclear Instructions', 'color': '#F59E0B', 'impact': 'Root'},
                    {'label': 'Patient Confusion', 'color': '#F59E0B', 'impact': 'Process'},
                    {'label': 'Negative Sentiment', 'color': '#EF4444', 'impact': 'Outcome'},
                ]
            }
        ]
    }

@st.cache_data
def generate_alerts_data():
    """Generate real-time alerts"""
    return [
        {
            'type': 'warning',
            'title': 'Spike in Billing Complaints',
            'description': '+23% increase in billing-related feedback in last 7 days',
            'time': '2 hours ago'
        },
        {
            'type': 'danger',
            'title': 'Wait Time Issues Escalating',
            'description': 'Critical: Wait time complaints increased by 18% week-over-week',
            'time': '4 hours ago'
        },
        {
            'type': 'warning',
            'title': 'Staff Communication Trend',
            'description': 'Negative sentiment in communication feedback trending up',
            'time': '6 hours ago'
        }
    ]

# ============================================================================
# NAVIGATION STATE
# ============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 CX-Intel")
    st.markdown("---")
    
    # Main navigation
    pages = [
        "Overview",
        "Cluster Analysis",
        "Sentiment Analysis",
        "Issue Prioritization",
        "Operational Impact",
        "Recommendations",
        "Real-time Alerts",
        "AI Assistant",
        "Reports",
        "Settings"
    ]
    
    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.session_state.page = selected_page
    
    st.markdown("---")
    st.markdown("### 🤖 AI Agents Powering CX-Intel")
    
    agents = [
        {"name": "Recommendation Agent", "icon": "💡"},
        {"name": "Priority Scoring Agent", "icon": "⚡"},
        {"name": "Cascade Agent", "icon": "🔗"},
        {"name": "Monitoring Agent", "icon": "👁️"},
        {"name": "Assistant Agent", "icon": "🎯"}
    ]
    
    for agent in agents:
        st.markdown(f"""
        <div class="sidebar-item">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{agent['icon']}</span>
            <span>{agent['name']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**Version**: 1.0.0  \n"
        "**Status**: Active  \n"
        "**Last Updated**: Today"
    )

# ============================================================================
# HEADER WITH FILTERS
# ============================================================================

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown("# 📈 Customer Feedback Intelligence Dashboard")

with col2:
    date_filter = st.selectbox(
        "Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"],
        label_visibility="collapsed"
    )

with col3:
    st.button("🔍 Filters", use_container_width=True)

st.markdown("---")

# ============================================================================
# LOAD DATA
# ============================================================================

data = generate_dashboard_data()
cascade_data = generate_cascade_data()
alerts_data = generate_alerts_data()

# ============================================================================
# PAGE: OVERVIEW (DEFAULT)
# ============================================================================

if st.session_state.page == "Overview":
    
    # KPI CARDS ROW
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    kpi_configs = [
        {
            'col': col1,
            'title': 'Total Feedback',
            'value': f"{data['total_feedback']:,}",
            'trend': '+12%',
            'trend_up': True,
            'color': '#3B82F6'
        },
        {
            'col': col2,
            'title': 'Negative Sentiment',
            'value': f"{(data['negative_count']/data['total_feedback']*100):.1f}%",
            'trend': '+3%',
            'trend_up': False,
            'color': '#EF4444'
        },
        {
            'col': col3,
            'title': 'High Priority Issues',
            'value': '23',
            'trend': '+5',
            'trend_up': False,
            'color': '#F59E0B'
        },
        {
            'col': col4,
            'title': 'Avg Response Time',
            'value': '4.2h',
            'trend': '-0.3h',
            'trend_up': True,
            'color': '#10B981'
        },
        {
            'col': col5,
            'title': 'Customer Satisfaction',
            'value': '78.5%',
            'trend': '+2.1%',
            'trend_up': True,
            'color': '#8B5CF6'
        }
    ]
    
    for config in kpi_configs:
        with config['col']:
            trend_class = 'trend-up' if config['trend_up'] else 'trend-down'
            trend_icon = '📈' if config['trend_up'] else '📉'
            
            st.markdown(f"""
            <div class="custom-card">
                <div class="kpi-label">{config['title']}</div>
                <div class="kpi-value" style="color: {config['color']};">{config['value']}</div>
                <div class="{trend_class}">{trend_icon} {config['trend']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MAIN GRID - ROW 1
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    # Sentiment Distribution (Donut Chart)
    with col1:
        st.markdown("### Sentiment Distribution")
        
        sentiment_values = [
            data['positive_count'],
            data['neutral_count'],
            data['negative_count']
        ]
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_colors = ['#10B981', '#6B7280', '#EF4444']
        
        fig_sentiment = go.Figure(data=[go.Pie(
            labels=sentiment_labels,
            values=sentiment_values,
            hole=.4,
            marker=dict(colors=sentiment_colors),
            textinfo='label+percent',
            textfont=dict(color='#F1F5F9', size=11),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_sentiment.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0.7,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(107, 114, 128, 0.3)",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
    
    # Top Issue Clusters (Horizontal Bar Chart)
    with col2:
        st.markdown("### Top Issue Clusters")
        
        issues_sorted = data['issues_df'].sort_values('Count', ascending=True).tail(7)
        
        fig_issues = go.Figure(data=[
            go.Bar(
                y=issues_sorted['Issue'],
                x=issues_sorted['Count'],
                orientation='h',
                marker=dict(
                    color=issues_sorted['Count'],
                    colorscale='Reds',
                    showscale=False
                ),
                text=issues_sorted['Count'],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>',
                textfont=dict(color='#F1F5F9', size=10)
            )
        ])
        
        fig_issues.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            margin=dict(l=150, r=50, t=0, b=0),
            height=350,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(107, 114, 128, 0.2)'),
            yaxis=dict(showgrid=False),
            showlegend=False,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_issues, use_container_width=True, config={'displayModeBar': False})
    
    # Priority Issues Panel
    with col3:
        st.markdown("### Priority Issues")
        
        priority_issues = [
            {
                'name': 'Wait Time / Delays',
                'desc': 'Long wait times affecting patient satisfaction',
                'priority': 'High',
                'count': 1850
            },
            {
                'name': 'Staff Communication',
                'desc': 'Communication gaps with patients',
                'priority': 'High',
                'count': 1420
            },
            {
                'name': 'Billing / Charges',
                'desc': 'Unexpected charges and billing issues',
                'priority': 'Medium',
                'count': 980
            }
        ]
        
        for issue in priority_issues:
            priority_badge = '<span class="badge-high">High</span>' if issue['priority'] == 'High' else '<span class="badge-medium">Medium</span>'
            
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; color: #F1F5F9;">{issue['name']}</h4>
                    {priority_badge}
                </div>
                <p class="insight-text">{issue['desc']}</p>
                <p style="font-size: 0.85rem; color: #3B82F6; font-weight: 600;">{issue['count']:,} mentions</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MAIN GRID - ROW 2
    col1, col2 = st.columns(2)
    
    # Sentiment Trend Over Time
    with col1:
        st.markdown("### Sentiment Trend (30 Days)")
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=data['sentiment_df']['date'],
            y=data['sentiment_df']['positive'],
            name='Positive',
            line=dict(color='#10B981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)',
            hovertemplate='<b>Positive</b><br>Date: %{x|%b %d}<br>Count: %{y:,}<extra></extra>'
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=data['sentiment_df']['date'],
            y=data['sentiment_df']['neutral'],
            name='Neutral',
            line=dict(color='#6B7280', width=3),
            fill='tozeroy',
            fillcolor='rgba(107, 114, 128, 0.1)',
            hovertemplate='<b>Neutral</b><br>Date: %{x|%b %d}<br>Count: %{y:,}<extra></extra>'
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=data['sentiment_df']['date'],
            y=data['sentiment_df']['negative'],
            name='Negative',
            line=dict(color='#EF4444', width=3),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',
            hovertemplate='<b>Negative</b><br>Date: %{x|%b %d}<br>Count: %{y:,}<extra></extra>'
        ))
        
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            margin=dict(l=50, r=20, t=0, b=0),
            height=350,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(107, 114, 128, 0.2)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(107, 114, 128, 0.2)'),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            )
        )
        
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    
    # AI Insights Summary
    with col2:
        st.markdown("### AI-Generated Insights")
        
        insights = [
            "🔴 Wait time issues are the primary driver of negative sentiment, accounting for 45% of complaints",
            "🟡 Communication gaps strongly correlate with low satisfaction scores (r=0.87)",
            "📈 Billing complaints are increasing rapidly at +23% week-over-week",
            "💡 Staff empathy training could reduce 18% of negative sentiment",
            "⚡ Implementing online queuing could reduce wait time complaints by ~35%",
            "✅ Cleanliness feedback is consistently positive (92% positive)"
        ]
        
        for insight in insights:
            st.markdown(f'<p class="insight-text">• {insight}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MAIN GRID - ROW 3
    col1, col2, col3 = st.columns(3)
    
    # Recommended Actions
    with col1:
        st.markdown("### Recommended Actions")
        
        actions = [
            {
                'title': 'Optimize Scheduling System',
                'desc': 'Implement AI-driven patient scheduling to reduce wait times',
                'impact': 'High'
            },
            {
                'title': 'Staff Training Program',
                'desc': 'Enhanced communication and empathy training for staff',
                'impact': 'High'
            },
            {
                'title': 'Billing Transparency',
                'desc': 'Clearer billing explanations and itemized receipts',
                'impact': 'Medium'
            }
        ]
        
        for action in actions:
            impact_badge = '<span class="badge-high">High</span>' if action['impact'] == 'High' else '<span class="badge-medium">Medium</span>'
            
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #F1F5F9;">{action['title']}</h4>
                <p class="insight-text">{action['desc']}</p>
                <div style="text-align: right; margin-top: 0.75rem;">Impact: {impact_badge}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Real-Time Alerts
    with col2:
        st.markdown("### Real-Time Alerts")
        
        for alert in alerts_data:
            alert_color = '#EF4444' if alert['type'] == 'danger' else '#F59E0B'
            
            st.markdown(f"""
            <div class="alert-box" style="border-left-color: {alert_color};">
                <h4 style="margin: 0 0 0.25rem 0; color: #F1F5F9;">{alert['title']}</h4>
                <p style="margin: 0.25rem 0; color: #CBD5E1; font-size: 0.9rem;">{alert['description']}</p>
                <p style="margin: 0.5rem 0 0 0; color: #6B7280; font-size: 0.8rem;">{alert['time']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Assistant
    with col3:
        st.markdown("### AI Assistant")
        st.markdown("#### Ask Questions About Your Data")
        
        st.text_input(
            "Enter your question",
            placeholder="E.g., 'What are the top 3 issues affecting satisfaction?'",
            label_visibility="collapsed"
        )
        
        st.markdown("**Example Prompts:**")
        example_prompts = [
            "📊 Compare wait time vs billing issues",
            "🔍 What drives negative sentiment?",
            "⚡ Show me actionable recommendations",
            "📈 Predict trends for next month",
            "💡 Root cause analysis for complaints"
        ]
        
        for prompt in example_prompts:
            st.markdown(f'<p class="insight-text">• {prompt}</p>', unsafe_allow_html=True)

# ============================================================================
# PAGE: CLUSTER ANALYSIS
# ============================================================================

elif st.session_state.page == "Cluster Analysis":
    st.markdown("## Cluster Analysis")
    st.info("Semantic clustering of feedback into topic groups")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cluster Distribution")
        
        clusters_data = {
            'Cluster': ['Wait Times', 'Communication', 'Billing', 'Quality', 'Facilities', 'Staff', 'Other'],
            'Size': [1850, 1420, 980, 756, 642, 538, 421]
        }
        
        fig = px.pie(
            x=clusters_data['Size'],
            labels=clusters_data['Cluster'],
            title="Distribution Across Clusters"
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("### Cluster Sentiment Analysis")
        
        cluster_sentiment = {
            'Cluster': clusters_data['Cluster'],
            'Negative': [1650, 980, 720, 540, 380, 290, 210],
            'Neutral': [150, 350, 200, 180, 200, 200, 150],
            'Positive': [50, 90, 60, 36, 62, 48, 61]
        }
        
        fig = go.Figure(data=[
            go.Bar(name='Negative', x=cluster_sentiment['Cluster'], y=cluster_sentiment['Negative'], marker_color='#EF4444'),
            go.Bar(name='Neutral', x=cluster_sentiment['Cluster'], y=cluster_sentiment['Neutral'], marker_color='#6B7280'),
            go.Bar(name='Positive', x=cluster_sentiment['Cluster'], y=cluster_sentiment['Positive'], marker_color='#10B981')
        ])
        
        fig.update_layout(
            barmode='stack',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: SENTIMENT ANALYSIS
# ============================================================================

elif st.session_state.page == "Sentiment Analysis":
    st.markdown("## Sentiment Analysis")
    st.info("Detailed sentiment classification and metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overall Sentiment Breakdown")
        
        sentiment_data = {
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [data['positive_count'], data['neutral_count'], data['negative_count']],
            'Percentage': [30.5, 47.7, 21.8]
        }
        
        df_sentiment = pd.DataFrame(sentiment_data)
        
        fig = go.Figure(data=[go.Bar(
            x=df_sentiment['Sentiment'],
            y=df_sentiment['Count'],
            marker_color=['#10B981', '#6B7280', '#EF4444'],
            text=df_sentiment['Count'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
        )])
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("### Sentiment Confidence Distribution")
        
        confidence_bins = np.linspace(0, 100, 11)
        confidence_data = np.random.normal(78, 12, 10000)
        
        fig = go.Figure(data=[go.Histogram(
            x=confidence_data,
            nbinsx=20,
            marker_color='#3B82F6',
            hovertemplate='<b>Confidence Range: %{x:.0f}%</b><br>Count: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
            height=400,
            xaxis_title="Confidence Score (%)",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: ISSUE PRIORITIZATION
# ============================================================================

elif st.session_state.page == "Issue Prioritization":
    st.markdown("## Issue Prioritization Matrix")
    st.info("Prioritize issues by impact and frequency")
    
    # Create prioritization matrix
    issues_priority = pd.DataFrame({
        'Issue': ['Wait Times', 'Communication', 'Billing', 'Service Quality', 'Cleanliness', 'Doctor Availability'],
        'Frequency': [1850, 1420, 980, 756, 642, 538],
        'Impact': [9.2, 8.5, 7.8, 7.5, 6.2, 6.8],
        'Sentiment_Score': [2.1, 1.8, 1.6, 1.4, 1.2, 1.5]
    })
    
    fig = go.Figure(data=go.Scatter(
        x=issues_priority['Frequency'],
        y=issues_priority['Impact'],
        mode='markers+text',
        text=issues_priority['Issue'],
        textposition="top center",
        marker=dict(
            size=issues_priority['Sentiment_Score']*15,
            color=issues_priority['Sentiment_Score'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Sentiment<br>Score"),
            line=dict(width=1, color='#CBD5E1')
        ),
        hovertemplate='<b>%{text}</b><br>Frequency: %{x:,}<br>Impact: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", color="#F1F5F9", size=11),
        height=500,
        xaxis_title="Frequency (Number of Mentions)",
        yaxis_title="Impact Score",
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("---")
    st.markdown("### Prioritization Methodology")
    st.markdown("""
    - **High Priority**: High frequency + High impact
    - **Medium Priority**: High frequency OR High impact
    - **Low Priority**: Low frequency + Low impact
    """)

# ============================================================================
# PAGE: OPERATIONAL IMPACT
# ============================================================================

elif st.session_state.page == "Operational Impact":
    st.markdown("## Operational Impact - Cause-Effect Cascades")
    
    for chain_idx, chain in enumerate(cascade_data['chains']):
        st.markdown(f"### {chain['title']}")
        
        # Create cascade flow visualization
        steps = chain['steps']
        
        # Create figure with annotation
        fig = go.Figure()
        
        x_positions = np.linspace(0, 3, len(steps))
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[1],
                mode='markers',
                marker=dict(
                    size=40,
                    color=step['color'],
                    opacity=0.8
                ),
                text=step['label'],
                textposition="bottom center",
                textfont=dict(size=10, color='#F1F5F9'),
                hovertemplate=f"<b>{step['label']}</b><br>Type: {step['impact']}<extra></extra>",
                showlegend=False
            ))
            
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=(x_positions[i] + x_positions[i+1]) / 2,
                    y=1,
                    text="→",
                    showarrow=False,
                    font=dict(size=20, color='#3B82F6'),
                    xanchor="center"
                )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#F1F5F9"),
            height=200,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(t=50, b=50),
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")

# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================

elif st.session_state.page == "Recommendations":
    st.markdown("## AI-Generated Recommendations")
    
    recommendations = [
        {
            'title': 'Implement Real-Time Queue Management System',
            'description': 'Deploy IoT sensors and mobile app integration to provide live wait time estimates and virtual queuing',
            'expected_impact': 'Reduce wait time complaints by 35-40%',
            'timeline': '2-3 months',
            'priority': 'High',
            'investment': 'Medium'
        },
        {
            'title': 'Enhanced Staff Communication Training',
            'description': 'Structured empathy and active listening training program with quarterly refresher sessions',
            'expected_impact': 'Improve communication sentiment by 25%',
            'timeline': '1-2 months',
            'priority': 'High',
            'investment': 'Low'
        },
        {
            'title': 'Automated Billing Clarity Initiative',
            'description': 'Implement AI-powered itemized receipt generation and insurance verification chatbot',
            'expected_impact': 'Reduce billing complaints by 30%',
            'timeline': '2-4 months',
            'priority': 'Medium',
            'investment': 'Medium'
        },
        {
            'title': 'Predictive Patient Satisfaction Monitoring',
            'description': 'Real-time ML model to identify at-risk patients and trigger proactive intervention protocols',
            'expected_impact': 'Improve overall satisfaction by 8-12%',
            'timeline': '3-4 months',
            'priority': 'Medium',
            'investment': 'High'
        },
    ]
    
    for rec in recommendations:
        priority_badge = '<span class="badge-high">HIGH</span>' if rec['priority'] == 'High' else '<span class="badge-medium">MEDIUM</span>'
        
        st.markdown(f"""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #F1F5F9; flex: 1;">{rec['title']}</h3>
                {priority_badge}
            </div>
            
            <p class="insight-text" style="margin-bottom: 1rem;">{rec['description']}</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Expected Impact</p>
                    <p style="margin: 0.25rem 0 0 0; color: #10B981; font-weight: 600;">{rec['expected_impact']}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Timeline</p>
                    <p style="margin: 0.25rem 0 0 0; color: #3B82F6; font-weight: 600;">{rec['timeline']}</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Priority</p>
                    <p style="margin: 0.25rem 0 0 0; color: #F1F5F9; font-weight: 600;">{rec['priority']}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Investment</p>
                    <p style="margin: 0.25rem 0 0 0; color: #F59E0B; font-weight: 600;">{rec['investment']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: REAL-TIME ALERTS
# ============================================================================

elif st.session_state.page == "Real-time Alerts":
    st.markdown("## Real-Time Monitoring & Alerts")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Alert Timeline")
        
        alerts_detailed = [
            {
                'time': '2:45 PM',
                'type': 'danger',
                'title': 'Critical: Wait Time Spike',
                'message': 'Wait time complaints increased by 18% (week-over-week)',
                'affected_area': 'Emergency Department'
            },
            {
                'time': '1:30 PM',
                'type': 'warning',
                'title': 'Billing Complaints Trend',
                'message': '+23% increase in billing-related feedback in last 7 days',
                'affected_area': 'Billing Department'
            },
            {
                'time': '12:15 PM',
                'type': 'warning',
                'title': 'Communication Issues',
                'message': 'Negative sentiment in communication feedback trending up',
                'affected_area': 'Multiple Departments'
            },
            {
                'time': '11:00 AM',
                'type': 'info',
                'title': 'Cleanliness Feedback Positive',
                'message': '92% positive feedback on facility cleanliness',
                'affected_area': 'Facility Management'
            }
        ]
        
        for alert in alerts_detailed:
            alert_color = '#EF4444' if alert['type'] == 'danger' else '#F59E0B' if alert['type'] == 'warning' else '#3B82F6'
            alert_icon = '🔴' if alert['type'] == 'danger' else '🟡' if alert['type'] == 'warning' else '🔵'
            
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; gap: 1rem;">
                    <div style="font-size: 1.5rem;">{alert_icon}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 0.25rem 0; color: #F1F5F9;">{alert['title']}</h4>
                        <p style="margin: 0.25rem 0; color: #CBD5E1; font-size: 0.9rem;">{alert['message']}</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.75rem; font-size: 0.85rem; color: #6B7280;">
                            <span>📍 {alert['affected_area']}</span>
                            <span>⏰ {alert['time']}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Alert Statistics")
        
        alert_stats = {
            'Total Alerts': '24',
            'Critical': '3',
            'Warning': '8',
            'Info': '13',
            'Resolved': '18'
        }
        
        for stat, value in alert_stats.items():
            st.markdown(f"""
            <div class="custom-card">
                <p class="kpi-label">{stat}</p>
                <p class="kpi-value">{value}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE: AI ASSISTANT
# ============================================================================

elif st.session_state.page == "AI Assistant":
    st.markdown("## AI Assistant")
    st.info("Ask questions about your feedback data and receive AI-powered insights")
    
    st.markdown("### Ask Your Questions")
    
    user_question = st.text_area(
        "Enter your question",
        placeholder="E.g., 'What are the top 3 root causes of negative sentiment?' or 'How has wait time sentiment changed over time?'",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze", use_container_width=True):
            st.success("Analysis complete!")
    
    with col2:
        st.button("📊 Visualize", use_container_width=True)
    
    with col3:
        st.button("💾 Save Query", use_container_width=True)
    
    if user_question:
        st.markdown("---")
        st.markdown("### AI Response")
        
        st.markdown("""
        Based on the analysis of 10,247 customer feedback entries:

        **Top 3 Root Causes of Negative Sentiment:**

        1. **Long Wait Times (45% of complaints)**
           - 1,850 mentions across all feedback
           - Strongly correlates with overall dissatisfaction (r=0.89)
           - Recommendation: Implement queue management system

        2. **Poor Staff Communication (32% of complaints)**
           - 1,420 mentions related to clarity and empathy
           - Affects patient trust and satisfaction
           - Recommendation: Communication skills training program

        3. **Unexpected Billing Charges (22% of complaints)**
           - 980 mentions of billing confusion
           - Trending upward at +23% weekly
           - Recommendation: Automated billing explanation system

        **Confidence Score:** 94.2%
        **Data Quality:** Excellent (99.68% parse success rate)
        """)

# ============================================================================
# PAGE: REPORTS
# ============================================================================

elif st.session_state.page == "Reports":
    st.markdown("## Reports & Export")
    
    st.markdown("### Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Available Reports")
        
        reports = [
            "Executive Summary (PDF)",
            "Detailed Sentiment Analysis",
            "Issue Prioritization Matrix",
            "Cascade Analysis Report",
            "Recommended Actions Plan"
        ]
        
        for report in reports:
            st.markdown(f"- 📄 {report}")
    
    with col2:
        st.markdown("#### Export Options")
        
        st.button("📥 Download Executive Summary (PDF)", use_container_width=True)
        st.button("📊 Export Data (CSV)", use_container_width=True)
        st.button("📈 Export Charts (PNG)", use_container_width=True)
        st.button("📋 Schedule Report Email", use_container_width=True)

# ============================================================================
# PAGE: SETTINGS
# ============================================================================

elif st.session_state.page == "Settings":
    st.markdown("## Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dashboard Settings")
        
        st.toggle("Dark Mode", value=True, disabled=True)
        st.toggle("Real-time Updates", value=True)
        st.toggle("Enable Notifications", value=True)
        
        st.markdown("### Data Settings")
        
        refresh_interval = st.select_slider(
            "Auto-refresh Interval",
            options=["5 min", "15 min", "30 min", "1 hour", "Manual"],
            value="30 min"
        )
        
        data_retention = st.selectbox(
            "Data Retention Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "Unlimited"]
        )
    
    with col2:
        st.markdown("### Alert Settings")
        
        alert_threshold = st.slider(
            "Alert Sensitivity (Lower = More Alerts)",
            0,
            100,
            50
        )
        
        st.markdown("### Account")
        
        st.text_input("Email", value="admin@cxintel.com", disabled=True)
        st.text_input("Organization", value="Hospital Network", disabled=True)
        st.selectbox("Role", ["Administrator", "Manager", "Analyst"])

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem; margin-top: 2rem;'>"
    "CX-Intel v1.0.0 | © 2024 Customer Experience Intelligence Platform | "
    "<a href='#' style='color: #3B82F6; text-decoration: none;'>Privacy Policy</a> | "
    "<a href='#' style='color: #3B82F6; text-decoration: none;'>Support</a>"
    "</div>",
    unsafe_allow_html=True
)
