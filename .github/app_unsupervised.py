"""
M4: Unsupervised-First Streamlit Dashboard
Interactive experience map with cluster auditing and causal insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_loader import DataLoader
from unsupervised_clustering import UnsupervisedClusteringEngine
from cluster_audit import ClusterAuditEngine
from causal_reasoning import CausalReasoningEngine
from agent.action_intelligence import CXActionIntelligenceAgent
from pathlib import Path

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CX-Intel Discovery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_app_theme():
    """Apply CX-Intel visual styling."""
    st.markdown(
        """
        <style>
        :root {
            --cx-blue-900: #0b1f3a;
            --cx-blue-700: #174ea6;
            --cx-blue-500: #2f7de1;
            --cx-blue-100: #e8f1ff;
            --cx-yellow-500: #f2c94c;
            --cx-yellow-100: #fff7d6;
            --cx-slate-700: #334155;
            --cx-slate-500: #64748b;
            --cx-border: rgba(23, 78, 166, 0.16);
            --cx-glass: rgba(255, 255, 255, 0.78);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(47, 125, 225, 0.18), transparent 34rem),
                linear-gradient(180deg, #f6f9ff 0%, #ffffff 44%, #f8fbff 100%);
            color: var(--cx-blue-900);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }

        .cx-hero {
            padding: 1.35rem 1.55rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(11, 31, 58, 0.94), rgba(23, 78, 166, 0.86)),
                linear-gradient(90deg, rgba(242, 201, 76, 0.18), transparent);
            box-shadow: 0 18px 45px rgba(23, 78, 166, 0.18);
        }

        .cx-hero h1 {
            margin: 0 0 0.35rem 0;
            color: #ffffff;
            font-size: 2.1rem;
            line-height: 1.15;
            font-weight: 760;
            letter-spacing: 0;
        }

        .cx-hero p {
            margin: 0;
            color: rgba(255, 255, 255, 0.86);
            font-size: 1rem;
        }

        .cx-agent-banner {
            padding: 0.85rem 1rem;
            margin: 0.35rem 0 1rem;
            border-left: 5px solid var(--cx-yellow-500);
            border-radius: 12px;
            background: rgba(255, 247, 214, 0.86);
            color: var(--cx-blue-900);
            box-shadow: 0 10px 28px rgba(11, 31, 58, 0.07);
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: 14px;
            background: var(--cx-glass);
            box-shadow: 0 12px 28px rgba(23, 78, 166, 0.08);
            backdrop-filter: blur(12px);
        }

        div[data-testid="stMetric"] label {
            color: var(--cx-slate-500);
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--cx-blue-900);
            font-weight: 760;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding: 0.35rem;
            border: 1px solid var(--cx-border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 10px 30px rgba(23, 78, 166, 0.08);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            color: var(--cx-slate-700);
            font-weight: 650;
        }

        .stTabs [aria-selected="true"] {
            background: var(--cx-blue-700);
            color: #ffffff;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--cx-border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 8px 24px rgba(23, 78, 166, 0.06);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--cx-border);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 12px 28px rgba(23, 78, 166, 0.07);
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid rgba(23, 78, 166, 0.22);
            border-radius: 10px;
            background: linear-gradient(180deg, #ffffff, #edf5ff);
            color: var(--cx-blue-900);
            font-weight: 700;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--cx-blue-500);
            color: var(--cx-blue-700);
            box-shadow: 0 8px 20px rgba(47, 125, 225, 0.16);
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--cx-border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 8px 20px rgba(23, 78, 166, 0.06);
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--cx-border);
        }

        h2, h3 {
            color: var(--cx-blue-900);
            letter-spacing: 0;
        }

        p, li, label {
            color: var(--cx-slate-700);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_app_theme()

st.markdown(
    """
    <section class="cx-hero">
        <h1>CX-Intel: Customer Experience Intelligence</h1>
        <p>Service feedback discovery, action intelligence, and AI-assisted decision support.</p>
    </section>
    <div class="cx-agent-banner">
        <strong>M5 AI Agent enabled.</strong>
        CX-Intel converts clusters, sentiment, keywords, root-cause signals, and cascades
        into prioritized management actions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Session State & Caching
# ============================================================================


@st.cache_resource
def load_data():
    """Load and normalize feedback data."""
    loader = DataLoader("text_data.csv")
    df, stats = loader.load()
    return df, stats


@st.cache_resource
def run_clustering():
    """Run unsupervised clustering pipeline."""
    df, stats = load_data()
    texts = df["text_normalized"].tolist()

    engine = UnsupervisedClusteringEngine(random_state=42)
    engine.fit(texts, auto_select=True)

    return engine, df, texts


@st.cache_resource
def run_audit(texts, cluster_assignments):
    """Run cluster auditing pipeline."""
    audit_engine = ClusterAuditEngine(model_device=-1)
    audit_engine.audit(texts, cluster_assignments, n_keywords=10)
    return audit_engine


@st.cache_resource
def run_causal_reasoning(
    cluster_lda_features, cluster_vocabularies, cluster_sentiments
):
    """Run causal reasoning pipeline."""
    causal_engine = CausalReasoningEngine(model_device=-1)
    causal_engine.reason(cluster_lda_features, cluster_vocabularies, cluster_sentiments)
    return causal_engine


# ============================================================================
# Main Dashboard
# ============================================================================


def main():
    """Main dashboard flow."""

    # Load data and run clustering
    with st.spinner("Loading data and discovering clusters..."):
        clustering_engine, df, texts = run_clustering()
        cluster_assignments = clustering_engine.cluster_assignments

    # Run audit
    with st.spinner("Auditing clusters (sentiment + vocabulary)..."):
        audit_engine = run_audit(texts, cluster_assignments)

    # Prepare data for causal reasoning
    cluster_lda_dict = {}
    for cid in np.unique(cluster_assignments):
        mask = cluster_assignments == cid
        cluster_lda_dict[cid] = clustering_engine.lda_features[mask]

    # Run causal reasoning
    with st.spinner("Analyzing causal relationships..."):
        causal_engine = run_causal_reasoning(
            cluster_lda_dict,
            audit_engine.cluster_vocabularies,
            audit_engine.cluster_sentiment_results,
        )

    # ====================================================================
    # Tab 1: Experience Map Visualization
    # ====================================================================

    action_agent = CXActionIntelligenceAgent()
    action_insights = action_agent.build_action_insights(
        audit_engine=audit_engine,
        causal_engine=causal_engine,
        clustering_engine=clustering_engine,
    )

    tab_action, tab_landscape, tab_audit, tab_causal, tab_data = st.tabs(
        [
            "🤖 AI Agent (M5)",
            "🗺️ Experience Map",
            "📊 Cluster Audit",
            "⚡ Operational Impact",
            "📋 Data Export",
        ]
    )

    # ====================================================================
    # Tab 0: Customer Experience Action Dashboard (M5)
    # ====================================================================

    with tab_action:
        st.markdown("### AI Agent Decision Support")
        st.caption(
            "M5 translates M1-M3 model outputs into practical next steps for service managers."
        )

        high_priority_count = sum(
            1 for insight in action_insights if insight.priority_label.startswith("HIGH")
        )
        medium_priority_count = sum(
            1
            for insight in action_insights
            if insight.priority_label.startswith("MEDIUM")
        )
        soft_cascade_count = sum(len(insight.cascades) for insight in action_insights)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Agent Insights", len(action_insights))
        with col2:
            st.metric("High Priority", high_priority_count)
        with col3:
            st.metric("Medium Priority", medium_priority_count)
        with col4:
            st.metric("Soft Cascades", soft_cascade_count)

        st.markdown("### Ask the CX Agent")
        st.caption(
            "Use the chat to turn the analysis into management decisions. "
            "It answers from the cluster insights already generated in this run."
        )

        if "cx_agent_messages" not in st.session_state:
            st.session_state.cx_agent_messages = [
                {
                    "role": "assistant",
                    "content": action_agent.answer_question(
                        "summarize for leadership", action_insights
                    ),
                }
            ]

        prompt_cols = st.columns(4)
        suggested_prompts = [
            "What should we fix first?",
            "Summarize this for leadership.",
            "Which cascades should we watch?",
            "What actions should managers take?",
        ]
        for col, prompt in zip(prompt_cols, suggested_prompts):
            with col:
                if st.button(prompt, key=f"agent_prompt_{prompt}"):
                    st.session_state.cx_agent_messages.append(
                        {"role": "user", "content": prompt}
                    )
                    st.session_state.cx_agent_messages.append(
                        {
                            "role": "assistant",
                            "content": action_agent.answer_question(
                                prompt, action_insights
                            ),
                        }
                    )

        for message in st.session_state.cx_agent_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input(
            "Ask about priorities, cascades, actions, or a specific cluster..."
        )
        if user_question:
            st.session_state.cx_agent_messages.append(
                {"role": "user", "content": user_question}
            )
            answer = action_agent.answer_question(user_question, action_insights)
            st.session_state.cx_agent_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

        st.markdown("---")
        st.markdown("### Customer Experience Action Dashboard")

        priority_filter = st.segmented_control(
            "Priority filter",
            options=["All", "HIGH 🔥", "MEDIUM ⚠", "LOW ✅"],
            default="All",
        )
        filtered_insights = (
            action_insights
            if priority_filter == "All"
            else [
                insight
                for insight in action_insights
                if insight.priority_label == priority_filter
            ]
        )

        action_df = action_agent.build_dashboard_dataframe(filtered_insights)
        st.dataframe(action_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Download Action Dashboard CSV",
            data=action_df.to_csv(index=False),
            file_name="cx_intel_action_dashboard.csv",
            mime="text/csv",
        )

        st.markdown("### Cluster Action Details")
        for insight in filtered_insights:
            with st.expander(
                f"AI Agent Insight | Cluster {insight.cluster_id}: {insight.issue_theme.title()} "
                f"| {insight.priority_label}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cluster Size", insight.metadata["cluster_size"])
                with col2:
                    st.metric(
                        "Negative Rate",
                        f"{insight.metadata['negative_rate']:.0%}",
                    )
                with col3:
                    st.metric("Priority Score", f"{insight.priority_score:.3f}")

                st.markdown("**Keywords**")
                st.write(", ".join(insight.keywords) if insight.keywords else "None")

                st.markdown("**Root Cause**")
                st.write(insight.root_cause)

                st.markdown("**Soft Cascades**")
                for cascade in insight.cascades:
                    st.write(f"- {cascade}")

                st.markdown("**Example Feedback**")
                st.info(insight.example_feedback)

                st.markdown("**Recommended Action**")
                st.success(insight.recommended_action)

    with tab_landscape:
        insight_by_cluster = {item.cluster_id: item for item in action_insights}
        high_clusters = [
            item for item in action_insights if item.priority_label.startswith("HIGH")
        ]
        top_agent_insight = action_insights[0] if action_insights else None

        st.markdown("### Agent Summary")
        st.info(action_agent.answer_question("summarize for leadership", action_insights))

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Feedback Entries", len(texts))
        with summary_col2:
            st.metric("Experience Clusters", clustering_engine.optimal_n_clusters)
        with summary_col3:
            st.metric("High-Priority Clusters", len(high_clusters))
        with summary_col4:
            st.metric(
                "Top Priority",
                (
                    f"Cluster {top_agent_insight.cluster_id}"
                    if top_agent_insight is not None
                    else "None"
                ),
            )

        st.markdown("---")
        st.markdown("### Experience Map")
        st.caption(
            "This map groups similar feedback near each other. Use the lens selector "
            "to switch between action priority, sentiment health, and discovered themes."
        )

        with st.expander("How to read this map"):
            st.markdown(
                """
                - Each dot is one feedback entry.
                - Dots close together discuss similar experience patterns.
                - Color changes based on the selected lens.
                - Larger dots mark high-priority clusters when using the priority lens.
                - Hover over a dot to see the AI Agent's theme, priority, action, and feedback snippet.
                """
            )

        col1, col2 = st.columns([3, 1])

        with col1:
            landscape_lens = st.radio(
                "Map lens",
                ["Priority Heatmap", "Sentiment Health", "Theme Clusters"],
                horizontal=True,
            )

            # Create the 2D experience map from the t-SNE projection.
            fig = go.Figure()

            priority_color_map = {
                "HIGH 🔥": "#D64545",
                "MEDIUM ⚠": "#F2A93B",
                "LOW ✅": "#2FA66A",
            }
            zone_color_map = {
                "RED_ZONE": "#FF6B6B",
                "GREEN_ZONE": "#51CF66",
                "NEUTRAL_ZONE": "#FFD93D",
            }
            cluster_palette = px.colors.qualitative.Safe

            for cluster_id in np.unique(cluster_assignments):
                mask = cluster_assignments == cluster_id
                sentiment_dist = audit_engine.cluster_sentiment_results[cluster_id][
                    "sentiment_distribution"
                ]
                zone = audit_engine.cluster_zones[cluster_id]["zone_type"]
                insight = insight_by_cluster.get(int(cluster_id))

                if landscape_lens == "Priority Heatmap" and insight is not None:
                    color = priority_color_map.get(insight.priority_label, "#999999")
                    trace_name = (
                        f"{insight.priority_label} | C{cluster_id} | "
                        f"{insight.issue_theme.title()}"
                    )
                elif landscape_lens == "Sentiment Health":
                    color = zone_color_map.get(zone, "#999999")
                    readable_zone = zone.replace("_", " ").title()
                    trace_name = f"C{cluster_id} | {readable_zone}"
                else:
                    color = cluster_palette[int(cluster_id) % len(cluster_palette)]
                    theme = insight.issue_theme.title() if insight else "Theme"
                    trace_name = f"C{cluster_id} | {theme}"

                hover_text = []
                for text in np.array(texts)[mask]:
                    if insight is not None:
                        hover_text.append(
                            f"<b>Cluster {cluster_id}</b><br>"
                            f"Theme: {insight.issue_theme.title()}<br>"
                            f"Priority: {insight.priority_label} "
                            f"({insight.priority_score:.3f})<br>"
                            f"Negative Feedback: {sentiment_dist['NEGATIVE']:.1%}<br>"
                            f"Action: {insight.recommended_action}<br><br>"
                            f"Feedback: {str(text)[:180]}"
                        )
                    else:
                        hover_text.append(
                            f"<b>Cluster {cluster_id}</b><br>"
                            f"Negative: {sentiment_dist['NEGATIVE']:.1%}<br>"
                            f"Feedback: {str(text)[:180]}"
                        )

                fig.add_trace(
                    go.Scatter(
                        x=clustering_engine.tsne_projection[mask, 0],
                        y=clustering_engine.tsne_projection[mask, 1],
                        mode="markers",
                        name=trace_name,
                        marker=dict(
                            size=(
                                9
                                if insight
                                and insight.priority_label.startswith("HIGH")
                                else 6
                            ),
                            color=color,
                            opacity=0.76,
                            line=dict(width=0.5, color="white"),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title=f"Experience Map - {landscape_lens}",
                xaxis_title="Experience Similarity Dimension 1",
                yaxis_title="Experience Similarity Dimension 2",
                hovermode="closest",
                height=560,
                width=800,
                legend_title="Map Legend",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Cluster Inspector")
            selected_map_cluster = st.selectbox(
                "Choose a cluster",
                [item.cluster_id for item in action_insights],
                format_func=lambda cid: (
                    f"Cluster {cid}: "
                    f"{insight_by_cluster[cid].issue_theme.title()} "
                    f"({insight_by_cluster[cid].priority_label})"
                ),
                key="landscape_agent_cluster",
            )
            selected_insight = insight_by_cluster[selected_map_cluster]

            st.markdown("**AI Agent Interpretation**")
            st.info(selected_insight.key_insight)

            st.markdown("**Recommended Action**")
            st.success(selected_insight.recommended_action)

            st.markdown("**Soft Cascades**")
            for cascade in selected_insight.cascades[:3]:
                st.write(f"- {cascade}")

            if st.button(
                f"Ask Agent about Cluster {selected_map_cluster}",
                key="landscape_ask_agent",
            ):
                prompt = f"Tell me about cluster {selected_map_cluster}"
                st.session_state.cx_agent_messages.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.cx_agent_messages.append(
                    {
                        "role": "assistant",
                        "content": action_agent.answer_question(
                            prompt, action_insights
                        ),
                    }
                )
                st.toast(
                    "Added the cluster question to the AI Agent chat tab.",
                    icon="🤖",
                )

            st.markdown("### Sentiment Health")
            red_count = len(audit_engine.get_red_zones())
            green_count = len(audit_engine.get_green_zones())
            neutral_count = len(audit_engine.get_neutral_zones())

            st.markdown(f"""
            🔴 **Distress clusters**: {red_count}  
            🟢 **Positive clusters**: {green_count}  
            🟡 **Mixed clusters**: {neutral_count}
            """)

    # ====================================================================
    # Tab 2: Cluster Auditing
    # ====================================================================

    with tab_audit:
        st.markdown("### Cluster Auditing Results")

        # Cluster selector
        selected_cluster = st.selectbox(
            "Select Cluster",
            sorted(audit_engine.cluster_texts.keys()),
            format_func=lambda x: f"Cluster {x} ({audit_engine.cluster_zones[x]['zone_type']})",
        )

        # Display audit report
        if selected_cluster in audit_engine.cluster_audit_reports:
            st.markdown(audit_engine.cluster_audit_reports[selected_cluster])

        # Display audit DataFrame
        st.markdown("### All Clusters Summary")
        audit_df = audit_engine.export_to_dataframe()
        st.dataframe(audit_df, use_container_width=True)

    # ====================================================================
    # Tab 3: Causal Analysis
    # ====================================================================

    with tab_causal:
        st.markdown("### Operational Impact Results")
        st.markdown(causal_engine.get_summary())

        cluster_options = sorted(causal_engine.cluster_lda_features.keys())
        if cluster_options:
            selected_causal_cluster = st.selectbox(
                "Select Cluster for Causal Details",
                cluster_options,
                format_func=lambda x: f"Cluster {x} ({audit_engine.cluster_zones.get(x, {}).get('zone_type', 'UNKNOWN')})",
            )

            st.markdown(causal_engine.get_cluster_summary(selected_causal_cluster))

            cascades = causal_engine.cascade_predictions.get(
                selected_causal_cluster, []
            )
            st.markdown(
                f"#### Cascade Predictions for Cluster {selected_causal_cluster}"
            )

            if cascades:
                for cascade in cascades:
                    st.info(f"""
                    **→ Cluster {cascade['target_cluster']}**  
                    Similarity: {cascade['similarity']:.1%}  
                    Cascade Likelihood: {cascade['cascade_likelihood']:.0%}  
                    {cascade['cascade_interpretation']}
                    """)
            else:
                st.info("No significant cascades detected for this cluster.")
        else:
            st.info("No causal clusters are available yet.")

        # Causal DataFrame
        st.markdown("### Operational Impact Summary")
        causal_df = causal_engine.export_to_dataframe()
        st.dataframe(causal_df, use_container_width=True)

    # ====================================================================
    # Tab 4: Data Export
    # ====================================================================

    with tab_data:
        st.markdown("### Export Enriched Data")

        # Combine all results into enriched dataframe
        export_df = clustering_engine.export_to_dataframe(texts, df)

        # Add audit results
        sentiment_data = pd.DataFrame(
            [
                {
                    "cluster_id": cid,
                    "sentiment_dominant": audit_engine.cluster_sentiment_results[cid][
                        "dominant_sentiment"
                    ],
                    "sentiment_density": audit_engine.cluster_sentiment_results[cid][
                        "sentiment_density"
                    ],
                    "zone_type": audit_engine.cluster_zones[cid]["zone_type"],
                }
                for cid in sorted(audit_engine.cluster_sentiment_results.keys())
            ]
        )

        # Create display
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Download Options")
            st.download_button(
                label="📥 Download Enriched CSV",
                data=export_df.to_csv(index=False),
                file_name="cx_intel_enriched.csv",
                mime="text/csv",
            )

            st.download_button(
                label="📥 Download Clustering Results (Pickle)",
                data=(
                    open("clustering_results.pkl", "rb").read()
                    if Path("clustering_results.pkl").exists()
                    else b""
                ),
                file_name="clustering_results.pkl",
                mime="application/octet-stream",
            )

        with col2:
            st.markdown("#### Data Preview")
            st.dataframe(export_df.head(10), use_container_width=True)

        st.markdown("#### Clustering Summary")
        st.info(clustering_engine.get_cluster_summary())


if __name__ == "__main__":
    main()
