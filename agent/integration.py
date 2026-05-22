"""
CX-Intel Agent Integration with Streamlit App
Adds agent-powered analysis tab to the existing UI.
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from agent.experiments import ExperimentPipeline
from agent.px_agent import PXIntelAgent
from agent.visualizations import create_agent_report_with_visualizations


def render_agent_tab(data_subset: pd.DataFrame, sentiment_model, deberta_model):
    """
    Render the Agent Analysis tab in Streamlit UI.

    Integration point: call this in app.py's main() function.
    """
    st.header("CX-Intel Agent Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analysis Options")
        run_agent = st.button("Run Agent Analysis", key="run_agent")
        run_experiments = st.checkbox("Run Experiment Pipeline", value=False)
        generate_visuals = st.checkbox("Generate Visualizations", value=False)

    with col2:
        st.subheader("About Agent")
        st.info(
            "The CX-Intel Agent automatically:\n"
            "1. Decides appropriate analysis depth\n"
            "2. Orchestrates sentiment + category + cause analysis\n"
            "3. Identifies priority issues for management\n"
            "4. Generates actionable recommendations"
        )

    st.markdown("---")

    if run_agent:
        agent = PXIntelAgent(sentiment_model=sentiment_model, deberta_model=deberta_model)

        with st.spinner("Agent analyzing feedback..."):
            texts = _get_texts(data_subset)
            insight = agent.orchestrate_analysis(data_subset, texts)

        st.success("Agent analysis complete!")
        _render_insight(agent, insight)

    if run_experiments:
        st.markdown("---")
        st.subheader("Experiment Pipeline")

        with st.spinner("Running 5 experiments..."):
            agent = PXIntelAgent(sentiment_model=sentiment_model, deberta_model=deberta_model)
            pipeline = ExperimentPipeline(agent, sentiment_model, deberta_model)
            results = pipeline.run_all_experiments(data_subset)

        st.success(f"Completed {len(results)} experiments!")

        for result in results:
            with st.expander(f"{result.experiment_id}: {result.experiment_name}"):
                if result.success:
                    st.write(result.insights)
                else:
                    st.error(f"Failed: {result.insights}")

        st.download_button(
            label="Download Experiment Summary",
            data=pipeline.generate_summary_report(),
            file_name="experiment_summary.txt",
            mime="text/plain",
        )

    if generate_visuals:
        st.markdown("---")
        st.subheader("Generate Visualizations")

        with st.spinner("Generating 5 visualizations..."):
            agent = PXIntelAgent(sentiment_model=sentiment_model, deberta_model=deberta_model)
            create_agent_report_with_visualizations(
                agent, data_subset, output_dir="agent_report"
            )

        st.success("Visualizations generated!")
        st.info("Saved to: `agent_report/` directory")

        viz_dir = Path("agent_report")
        if viz_dir.exists():
            images = sorted(viz_dir.glob("*.png"))
            if images:
                st.subheader("Generated Charts")
                for img_path in images:
                    st.image(str(img_path))


def _render_insight(agent: PXIntelAgent, insight):
    """Render an AgentInsight in Streamlit."""
    st.subheader("Analysis Metadata")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analysis Type", insight.analysis_type.value.upper())
    with col2:
        st.metric("Total Entries", insight.total_entries)
    with col3:
        st.metric("Timestamp", insight.timestamp.split("T")[0])
    with col4:
        st.metric("Priority Issues", len(insight.priority_issues))

    st.markdown("---")
    st.subheader("Sentiment Distribution")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Positive",
            insight.sentiment_summary["positive_count"],
            f"{insight.sentiment_summary['positive_rate']}%",
        )
    with col2:
        st.metric(
            "Negative",
            insight.sentiment_summary["negative_count"],
            f"{insight.sentiment_summary['negative_rate']}%",
        )
    with col3:
        st.metric(
            "Neutral",
            insight.sentiment_summary["neutral_count"],
            f"{insight.sentiment_summary['neutral_rate']}%",
        )

    st.markdown("---")
    st.subheader("Priority Issues Detected by Agent")

    if insight.priority_issues:
        for issue in insight.priority_issues[:5]:
            with st.expander(
                f"[{issue.rank}] {issue.issue_type.upper()} - "
                f"{issue.severity.value.upper()} ({issue.affected_entries} entries)"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Affected Entries", issue.affected_entries)
                with col2:
                    st.metric("Confidence", f"{issue.confidence:.1%}")
                with col3:
                    st.metric("Avg Sentiment", f"{issue.avg_sentiment_score:.2f}")

                st.markdown("**Recommended Action:**")
                st.write(issue.recommended_action)
    else:
        st.info("No priority issues detected. Feedback sentiment is generally positive.")

    st.markdown("---")
    st.subheader("Management Recommendations")

    if insight.recommendations:
        for i, rec in enumerate(insight.recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.info("No urgent recommendations at this time.")

    st.markdown("---")
    with st.expander("Full Agent Report"):
        st.text(agent.generate_report())

    st.markdown("---")
    st.subheader("Download Report")
    report_text = agent.generate_report()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="cx_intel_agent_report.txt",
            mime="text/plain",
        )

    with col2:
        report_json = json.dumps(
            {
                "timestamp": insight.timestamp,
                "analysis_type": insight.analysis_type.value,
                "total_entries": insight.total_entries,
                "sentiment_summary": insight.sentiment_summary,
                "priority_issues": [
                    {
                        "rank": issue.rank,
                        "issue_type": issue.issue_type,
                        "severity": issue.severity.value,
                        "frequency": issue.frequency,
                        "confidence": issue.confidence,
                    }
                    for issue in insight.priority_issues
                ],
                "recommendations": insight.recommendations,
            },
            indent=2,
        )

        st.download_button(
            label="Download Data (JSON)",
            data=report_json,
            file_name="cx_intel_agent_data.json",
            mime="application/json",
        )


def _get_texts(df: pd.DataFrame):
    """Return the most likely text column as a plain string list."""
    for column in ("content", "text_normalized", "text", "feedback"):
        if column in df.columns:
            return df[column].fillna("").astype(str).tolist()
    return []
