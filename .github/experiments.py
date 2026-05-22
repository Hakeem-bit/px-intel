"""
CX-Intel Agent Experiment Pipeline
5 validation experiments for agent system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class ExperimentResult:
    """Result from a single experiment."""

    experiment_id: str
    experiment_name: str
    success: bool
    insights: str
    metrics: Dict[str, Any]


class ExperimentPipeline:
    """Run validation experiments on agent system."""

    def __init__(self, agent, sentiment_model=None, deberta_model=None):
        """
        Initialize experiment pipeline.

        Args:
            agent: PXIntelAgent instance.
            sentiment_model: Optional RoBERTa model.
            deberta_model: Optional DeBERTa model.
        """
        self.agent = agent
        self.sentiment_model = sentiment_model
        self.deberta_model = deberta_model
        self.results: List[ExperimentResult] = []

    def run_all_experiments(self, df: pd.DataFrame) -> List[ExperimentResult]:
        """Run all 5 experiments."""
        self.results = [
            self._exp_baseline_vs_agent(df),
            self._exp_dataset_scaling(df),
            self._exp_crisis_detection(df),
            self._exp_category_imbalance(df),
            self._exp_root_cause_accuracy(df),
        ]
        return self.results

    def _get_texts(self, df: pd.DataFrame) -> List[str]:
        """Return the most likely text column as a plain string list."""
        for column in ("content", "text_normalized", "text", "feedback"):
            if column in df.columns:
                return df[column].fillna("").astype(str).tolist()
        return []

    def _exp_baseline_vs_agent(self, df: pd.DataFrame) -> ExperimentResult:
        """
        EXP-001: Baseline sentiment-only analysis vs full agent analysis.

        Objective: Demonstrate value of intelligent analysis.
        Expected: Agent provides 3-5x more insights.
        """
        try:
            texts = self._get_texts(df)

            baseline_insights = 1
            insight = self.agent.orchestrate_analysis(df, texts)

            agent_insights = len(insight.priority_issues) + len(insight.recommendations)
            multiplier = agent_insights / max(baseline_insights, 1)
            success = multiplier >= 3.0

            insights_text = f"""
Baseline provides: {baseline_insights} insight (sentiment distribution)
Agent provides: {agent_insights} insights (issues + recommendations)
Multiplier: {multiplier:.1f}x
Result: {'PASS' if success else 'FAIL'} (threshold: 3.0x)
            """

            return ExperimentResult(
                experiment_id="EXP-001",
                experiment_name="Baseline vs Agent",
                success=success,
                insights=insights_text,
                metrics={
                    "baseline_insights": baseline_insights,
                    "agent_insights": agent_insights,
                    "multiplier": multiplier,
                },
            )
        except Exception as exc:
            return self._error_result("EXP-001", "Baseline vs Agent", exc)

    def _exp_dataset_scaling(self, df: pd.DataFrame) -> ExperimentResult:
        """
        EXP-002: Dataset Scaling.

        Objective: Verify decision logic adapts to dataset size.
        Expected: Correct AnalysisType selected for each size.
        """
        try:
            results = {}

            small_type = self.agent.decide_analysis_type(df.head(30))
            results["small"] = small_type.value

            medium_type = self.agent.decide_analysis_type(df.head(min(100, len(df))))
            results["medium"] = medium_type.value

            large_type = self.agent.decide_analysis_type(df.head(min(300, len(df))))
            results["large"] = large_type.value

            success = (
                small_type.value == "sentiment_only"
                and medium_type.value in {"full_analysis", "sentiment_only"}
                and large_type.value in {"priority_detection", "full_analysis"}
            )

            insights_text = f"""
Small (30 entries): {small_type.value}
Medium (100 entries): {medium_type.value}
Large (300+ entries): {large_type.value}
Result: {'PASS' if success else 'FAIL'} (decision logic adapts correctly)
            """

            return ExperimentResult(
                experiment_id="EXP-002",
                experiment_name="Dataset Scaling",
                success=success,
                insights=insights_text,
                metrics=results,
            )
        except Exception as exc:
            return self._error_result("EXP-002", "Dataset Scaling", exc)

    def _exp_crisis_detection(self, df: pd.DataFrame) -> ExperimentResult:
        """
        EXP-003: Crisis Detection.

        Objective: Test urgent response capability.
        Expected: Triggers crisis mode at >40% negative.
        """
        try:
            texts = self._get_texts(df)
            insight = self.agent.orchestrate_analysis(df, texts)

            negative_rate = insight.sentiment_summary["negative_rate"]
            is_crisis = insight.analysis_type.value == "crisis_mode"
            should_be_crisis = negative_rate > 40
            success = is_crisis == should_be_crisis

            insights_text = f"""
Negative sentiment rate: {negative_rate}%
Analysis type: {insight.analysis_type.value}
Crisis detected: {is_crisis}
Should be crisis (>40%): {should_be_crisis}
Result: {'PASS' if success else 'FAIL'} (crisis detection works)
            """

            return ExperimentResult(
                experiment_id="EXP-003",
                experiment_name="Crisis Detection",
                success=success,
                insights=insights_text,
                metrics={
                    "negative_rate": negative_rate,
                    "is_crisis": is_crisis,
                    "should_be_crisis": should_be_crisis,
                },
            )
        except Exception as exc:
            return self._error_result("EXP-003", "Crisis Detection", exc)

    def _exp_category_imbalance(self, df: pd.DataFrame) -> ExperimentResult:
        """
        EXP-004: Category Imbalance Detection.

        Objective: Identify dominant issue patterns.
        Expected: Top issue affects >25% of entries.
        """
        try:
            texts = self._get_texts(df)
            insight = self.agent.orchestrate_analysis(df, texts)

            top_issue_pct = 0.0
            if insight.priority_issues:
                top_issue = insight.priority_issues[0]
                top_issue_pct = 100 * top_issue.affected_entries / max(1, len(df))
                success = top_issue_pct > 25

                insights_text = f"""
Top issue: {top_issue.issue_type} ({top_issue.severity.value})
Affected entries: {top_issue.affected_entries} / {len(df)}
Percentage: {top_issue_pct:.1f}%
Result: {'PASS' if success else 'FAIL'} (imbalance detected)
                """
            else:
                success = False
                insights_text = "No priority issues detected"

            return ExperimentResult(
                experiment_id="EXP-004",
                experiment_name="Category Imbalance",
                success=success,
                insights=insights_text,
                metrics={"top_issue_percentage": top_issue_pct},
            )
        except Exception as exc:
            return self._error_result("EXP-004", "Category Imbalance", exc)

    def _exp_root_cause_accuracy(self, df: pd.DataFrame) -> ExperimentResult:
        """
        EXP-005: Root-Cause Accuracy.

        Objective: Validate cause-effect correlations.
        Expected: Correlation score >0.5.
        """
        try:
            texts = self._get_texts(df)
            insight = self.agent.orchestrate_analysis(df, texts)

            if insight.cause_summary:
                avg_cause_score = float(np.mean(list(insight.cause_summary.values())))
                success = avg_cause_score > 0.5
            else:
                avg_cause_score = 0.0
                success = False

            identified_causes = (
                ", ".join(list(insight.cause_summary.keys())[:3])
                if insight.cause_summary
                else "None"
            )
            insights_text = f"""
Cause summary entries: {len(insight.cause_summary)}
Average cause score: {avg_cause_score:.3f}
Result: {'PASS' if success else 'FAIL'} (threshold: >0.5)
Identified causes: {identified_causes}
            """

            return ExperimentResult(
                experiment_id="EXP-005",
                experiment_name="Root-Cause Accuracy",
                success=success,
                insights=insights_text,
                metrics={
                    "avg_cause_score": avg_cause_score,
                    "num_causes_identified": len(insight.cause_summary),
                },
            )
        except Exception as exc:
            return self._error_result("EXP-005", "Root-Cause Accuracy", exc)

    def generate_summary_report(self) -> str:
        """Generate text summary of all experiments."""
        if not self.results:
            return "No experiments run yet."

        report = """
CX-INTEL AGENT EXPERIMENT SUMMARY
==================================

"""

        passed = sum(1 for result in self.results if result.success)
        total = len(self.results)
        report += f"Overall: {passed}/{total} experiments passed\n\n"

        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            report += f"{result.experiment_id}: {result.experiment_name} {status}\n"
            report += result.insights + "\n"

        return report

    def save_results(self, output_dir: str = "experiment_results") -> Path:
        """Save experiment results to a summary text file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / "experiment_summary.txt"
        report_file.write_text(self.generate_summary_report(), encoding="utf-8")
        return report_file

    @staticmethod
    def _error_result(
        experiment_id: str, experiment_name: str, exc: Exception
    ) -> ExperimentResult:
        """Build a standard failed experiment result for exceptions."""
        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            success=False,
            insights=f"Error: {exc}",
            metrics={},
        )
