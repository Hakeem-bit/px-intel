"""
CX-Intel Agent Visualization Module
Generates 5 publication-quality charts.
"""

from pathlib import Path
import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_cache_dir = Path(tempfile.gettempdir()) / "cx_intel_matplotlib"
_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))


def create_agent_report_with_visualizations(
    agent, df: pd.DataFrame, output_dir: str = "agent_report"
) -> Tuple[Any, Any]:
    """
    Create complete agent report with 5 visualizations.

    Args:
        agent: PXIntelAgent instance.
        df: Feedback dataframe.
        output_dir: Output directory for PNG files.

    Returns:
        Tuple of (insight, visualizer).
    """
    texts = _get_texts(df)
    insight = agent.orchestrate_analysis(df, texts)

    visualizer = AgentVisualizer(output_dir=output_dir)
    visualizer.generate_all_visualizations(
        df=df,
        sentiment_summary=insight.sentiment_summary,
        category_summary=insight.category_summary,
        cause_summary=insight.cause_summary,
        priority_issues=insight.priority_issues,
    )

    return insight, visualizer


class AgentVisualizer:
    """Generate publication-quality visualizations."""

    def __init__(self, output_dir: str = "agent_report"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_all_visualizations(
        self,
        df: pd.DataFrame,
        sentiment_summary: Dict,
        category_summary: Dict,
        cause_summary: Dict,
        priority_issues: List,
    ):
        """Generate all 5 visualizations."""
        self._viz_sentiment_distribution(sentiment_summary)
        self._viz_category_frequency(category_summary)
        self._viz_cause_distribution(cause_summary)
        self._viz_priority_ranking(priority_issues)
        self._viz_before_after_comparison(sentiment_summary, priority_issues)

    def _viz_sentiment_distribution(self, sentiment_summary: Dict):
        """VIZ 1: Sentiment distribution bar chart."""
        plt, _ = _load_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 6))

        sentiments = ["Positive", "Negative", "Neutral"]
        counts = [
            sentiment_summary["positive_count"],
            sentiment_summary["negative_count"],
            sentiment_summary["neutral_count"],
        ]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

        bars = ax.bar(
            sentiments,
            counts,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_ylabel("Number of Entries", fontsize=12, fontweight="bold")
        ax.set_title("Sentiment Distribution Analysis", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "01_sentiment_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _viz_category_frequency(self, category_summary: Dict):
        """VIZ 2: Category frequency horizontal bar chart."""
        plt, _ = _load_matplotlib()
        if not category_summary:
            category_summary = {"No data": 0}

        fig, ax = plt.subplots(figsize=(10, 6))

        sorted_pairs = sorted(
            category_summary.items(), key=lambda item: item[1], reverse=True
        )
        categories, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

        colors_map = {
            "wait times": "#e74c3c",
            "staff behavior": "#e67e22",
            "cleanliness": "#f39c12",
            "treatment quality": "#3498db",
            "costs": "#9b59b6",
            "general": "#95a5a6",
        }
        colors = [colors_map.get(cat, "#3498db") for cat in categories]

        bars = ax.barh(
            categories,
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        ax.set_xlabel("Confidence Score", fontsize=12, fontweight="bold")
        ax.set_title("Issue Category Frequency", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "02_category_frequency.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _viz_cause_distribution(self, cause_summary: Dict):
        """VIZ 3: Root cause distribution pie chart."""
        plt, _ = _load_matplotlib()
        if not cause_summary or sum(cause_summary.values()) == 0:
            cause_summary = {"No causes identified": 1}

        fig, ax = plt.subplots(figsize=(10, 8))

        causes = list(cause_summary.keys())
        values = list(cause_summary.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(causes)))

        wedges, texts, autotexts = ax.pie(
            values,
            labels=causes,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        _ = wedges, texts

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(11)
            autotext.set_fontweight("bold")

        ax.set_title(
            "Root Cause Distribution (Negative Feedback)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "03_root_cause_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _viz_priority_ranking(self, priority_issues: List):
        """VIZ 4: Priority issue ranking."""
        plt, mpatches = _load_matplotlib()
        fig, ax = plt.subplots(figsize=(12, 7))
        top_issues = priority_issues[:8]

        if not top_issues:
            ax.text(
                0.5,
                0.5,
                "No priority issues detected",
                ha="center",
                va="center",
            )
            plt.savefig(
                self.output_dir / "04_priority_ranking.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return

        issue_names = [f"#{issue.rank}: {issue.issue_type.title()}" for issue in top_issues]
        affected_counts = [issue.affected_entries for issue in top_issues]
        severity_colors = {
            "critical": "#e74c3c",
            "high": "#e67e22",
            "medium": "#f39c12",
            "low": "#2ecc71",
        }
        colors = [
            severity_colors.get(issue.severity.value, "#3498db")
            for issue in top_issues
        ]

        y_pos = np.arange(len(issue_names))
        bars = ax.barh(
            y_pos,
            affected_counts,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )

        for bar, issue in zip(bars, top_issues):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)} entries ({issue.severity.value})",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(issue_names, fontsize=11)
        ax.set_xlabel("Number of Affected Entries", fontsize=12, fontweight="bold")
        ax.set_title("Agent Priority Issues Ranking", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        legend_elements = [
            mpatches.Patch(color="#e74c3c", label="Critical"),
            mpatches.Patch(color="#e67e22", label="High"),
            mpatches.Patch(color="#f39c12", label="Medium"),
            mpatches.Patch(color="#2ecc71", label="Low"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "04_priority_ranking.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _viz_before_after_comparison(
        self, sentiment_summary: Dict, priority_issues: List
    ):
        """VIZ 5: Before vs After comparison."""
        plt, _ = _load_matplotlib()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sentiments_before = ["Positive", "Negative", "Neutral"]
        counts_before = [
            sentiment_summary["positive_count"],
            sentiment_summary["negative_count"],
            sentiment_summary["neutral_count"],
        ]
        colors_before = ["#2ecc71", "#e74c3c", "#95a5a6"]

        ax1.bar(
            sentiments_before,
            counts_before,
            color=colors_before,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax1.set_title(
            "BEFORE: Baseline Analysis\n(Sentiment Only)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_ylabel("Count", fontsize=10)
        ax1.grid(axis="y", alpha=0.3)

        for i, value in enumerate(counts_before):
            ax1.text(i, value, str(int(value)), ha="center", va="bottom", fontweight="bold")

        metrics = ["Priority\nIssues", "High/Critical\nSeverity", "Recommendations"]
        num_issues = len(priority_issues)
        num_critical = sum(
            1
            for issue in priority_issues
            if issue.severity.value in {"critical", "high"}
        )
        num_recommendations = len(priority_issues) if priority_issues else 0

        counts_after = [num_issues, num_critical, num_recommendations]
        colors_after = ["#3498db", "#e74c3c", "#2ecc71"]

        ax2.bar(
            metrics,
            counts_after,
            color=colors_after,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax2.set_title(
            "AFTER: Agent Analysis\n(Sentiment + Categories + Causes)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_ylabel("Count", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        for i, value in enumerate(counts_after):
            ax2.text(i, value, str(int(value)), ha="center", va="bottom", fontweight="bold")

        fig.suptitle(
            "Agent Value Comparison: Baseline vs Intelligent Analysis",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "05_before_after_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def _get_texts(df: pd.DataFrame) -> List[str]:
    """Return the most likely text column as a plain string list."""
    for column in ("content", "text_normalized", "text", "feedback"):
        if column in df.columns:
            return df[column].fillna("").astype(str).tolist()
    return []


def _load_matplotlib():
    """Load Matplotlib only when chart generation is requested."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    return plt, mpatches
