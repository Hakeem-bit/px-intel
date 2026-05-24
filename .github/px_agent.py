"""
CX-Intel Agent Core Logic
Intelligent orchestration of sentiment analysis, categorization, and causal inference.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class AnalysisType(Enum):
    """Analysis depth selection."""

    SENTIMENT_ONLY = "sentiment_only"
    FULL_ANALYSIS = "full_analysis"
    PRIORITY_DETECTION = "priority_detection"
    CRISIS_MODE = "crisis_mode"


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PriorityIssue:
    """Represents a priority issue detected by the agent."""

    rank: int
    issue_type: str
    severity: Severity
    frequency: int
    affected_entries: int
    confidence: float
    avg_sentiment_score: float
    recommended_action: str


@dataclass
class AgentInsight:
    """Complete agent analysis output."""

    timestamp: str
    analysis_type: AnalysisType
    total_entries: int
    sentiment_summary: Dict[str, Any]
    category_summary: Dict[str, float]
    cause_summary: Dict[str, float]
    priority_issues: List[PriorityIssue]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PXIntelAgent:
    """
    Intelligent orchestration agent for customer experience feedback analysis.

    Decision Logic:
    - <50 entries: SENTIMENT_ONLY
    - 50-200 entries: FULL_ANALYSIS
    - >200 entries: PRIORITY_DETECTION
    - >40% negative: CRISIS_MODE
    """

    def __init__(self, sentiment_model, deberta_model):
        """
        Initialize agent with models.

        Args:
            sentiment_model: RoBERTa sentiment classification pipeline.
            deberta_model: DeBERTa zero-shot classification pipeline.
        """
        self.sentiment_model = sentiment_model
        self.deberta_model = deberta_model
        self.last_insight: Optional[AgentInsight] = None

        self.action_map = {
            "wait times": "URGENT: Implement queue management system or appointment scheduling",
            "staff behavior": "PRIORITY: Conduct staff communication training and empathy workshops",
            "cleanliness": "MONITOR: Schedule facility maintenance and cleanliness audit",
            "treatment quality": "IMPORTANT: Audit clinical protocols and treatment standards",
            "costs": "FOLLOW-UP: Review billing procedures and cost transparency",
            "general": "OBSERVE: Monitor feedback trends over time",
        }

    def decide_analysis_type(self, df: pd.DataFrame) -> AnalysisType:
        """
        Decide what type of analysis to run based on dataset characteristics.

        Rules:
        - <50 entries: SENTIMENT_ONLY
        - 50-200 entries: FULL_ANALYSIS
        - >200 entries: PRIORITY_DETECTION
        - >40% negative: CRISIS_MODE override when sentiment is available.
        """
        n_entries = len(df)

        if n_entries and "sentiment" in df.columns:
            sentiments = df["sentiment"].astype(str).str.upper()
            negative_pct = (sentiments == "NEGATIVE").sum() / n_entries
            if negative_pct > 0.40:
                return AnalysisType.CRISIS_MODE

        if n_entries < 50:
            return AnalysisType.SENTIMENT_ONLY
        if n_entries < 200:
            return AnalysisType.FULL_ANALYSIS
        return AnalysisType.PRIORITY_DETECTION

    def orchestrate_analysis(self, df: pd.DataFrame, texts: List[str]) -> AgentInsight:
        """
        Run complete analysis pipeline: sentiment, categories, causes, priorities.

        Args:
            df: DataFrame with feedback data.
            texts: List of feedback texts.

        Returns:
            AgentInsight with complete analysis.
        """
        analysis_type = self.decide_analysis_type(df)
        sentiment_results = self._sentiment_analysis(texts)
        sentiment_summary = self._summarize_sentiment(sentiment_results)

        if sentiment_summary["negative_rate"] > 40:
            analysis_type = AnalysisType.CRISIS_MODE

        deep_analysis_types = {
            AnalysisType.FULL_ANALYSIS,
            AnalysisType.PRIORITY_DETECTION,
            AnalysisType.CRISIS_MODE,
        }
        if analysis_type in deep_analysis_types:
            category_results = self._category_analysis(texts)
            cause_results = self._cause_analysis(texts, sentiment_results)
        else:
            category_results = {}
            cause_results = {}

        priority_issues = self._detect_priority_issues(
            sentiment_results,
            category_results,
            cause_results,
            texts,
        )
        recommendations = self._generate_recommendations(
            priority_issues, sentiment_summary
        )

        insight = AgentInsight(
            timestamp=datetime.now().isoformat(),
            analysis_type=analysis_type,
            total_entries=len(df),
            sentiment_summary=sentiment_summary,
            category_summary=self._summarize_dict(category_results),
            cause_summary=self._summarize_dict(cause_results),
            priority_issues=priority_issues,
            recommendations=recommendations,
            metadata={
                "model_sentiment": "RoBERTa",
                "model_classification": "DeBERTa",
                "confidence_threshold": 0.5,
            },
        )

        self.last_insight = insight
        return insight

    def _sentiment_analysis(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run sentiment analysis on texts."""
        results = []
        for text in texts:
            try:
                if not text or len(str(text).strip()) < 5:
                    results.append({"label": "NEUTRAL", "score": 0.5})
                    continue

                text_truncated = str(text)[:512]
                result = self.sentiment_model(
                    text_truncated, truncation=True, max_length=512
                )
                results.append(
                    {
                        "label": self._normalize_sentiment_label(result[0]["label"]),
                        "score": float(result[0]["score"]),
                    }
                )
            except Exception:
                results.append({"label": "NEUTRAL", "score": 0.5})

        return results

    def _category_analysis(self, texts: List[str]) -> Dict[str, List[float]]:
        """Categorize feedback into issue types."""
        hypothesis_to_category = {
            "This feedback mentions wait times or scheduling issues": "wait times",
            "This feedback mentions staff behavior or communication": "staff behavior",
            "This feedback mentions cleanliness or facility conditions": "cleanliness",
            "This feedback mentions treatment quality or outcomes": "treatment quality",
            "This feedback mentions costs or billing issues": "costs",
            "This feedback is general feedback or other issues": "general",
        }

        results: Dict[str, List[float]] = {}
        hypotheses = list(hypothesis_to_category.keys())

        for text in texts:
            try:
                if not text or len(str(text).strip()) < 10:
                    continue

                result = self.deberta_model(
                    str(text)[:512],
                    hypotheses,
                    multi_label=True,
                    truncation=True,
                    max_length=512,
                )

                for label, score in zip(result["labels"], result["scores"]):
                    cat = hypothesis_to_category.get(label, "general")
                    results.setdefault(cat, []).append(float(score))
            except TypeError:
                result = self.deberta_model(
                    str(text)[:512],
                    hypotheses,
                    multi_class=True,
                    truncation=True,
                    max_length=512,
                )
                for label, score in zip(result["labels"], result["scores"]):
                    cat = hypothesis_to_category.get(label, "general")
                    results.setdefault(cat, []).append(float(score))
            except Exception:
                continue

        return results

    def _cause_analysis(
        self, texts: List[str], sentiment_results: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Analyze root causes for negative feedback."""
        hypothesis_to_cause = {
            "The reason for dissatisfaction is inadequate staffing": "inadequate staffing",
            "The reason for dissatisfaction is poor communication": "poor communication",
            "The reason for dissatisfaction is facility maintenance issues": "facility maintenance",
            "The reason for dissatisfaction is long wait times": "long wait times",
            "The reason for dissatisfaction is lack of empathy or care": "lack of empathy",
        }

        results: Dict[str, List[float]] = {}
        hypotheses = list(hypothesis_to_cause.keys())

        for text, sentiment in zip(texts, sentiment_results):
            if sentiment.get("label") != "NEGATIVE":
                continue

            try:
                result = self.deberta_model(
                    str(text)[:512],
                    hypotheses,
                    multi_label=True,
                    truncation=True,
                    max_length=512,
                )
            except TypeError:
                result = self.deberta_model(
                    str(text)[:512],
                    hypotheses,
                    multi_class=True,
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                continue

            for label, score in zip(result["labels"], result["scores"]):
                if score >= 0.40:
                    cause = hypothesis_to_cause.get(label, "unknown")
                    results.setdefault(cause, []).append(float(score))

        return results

    def _summarize_sentiment(
        self, sentiment_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize sentiment analysis results."""
        if not sentiment_results:
            return {
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_rate": 0,
                "negative_rate": 0,
                "neutral_rate": 0,
            }

        labels = [self._normalize_sentiment_label(r.get("label", "NEUTRAL")) for r in sentiment_results]
        total = len(labels)

        positive = labels.count("POSITIVE")
        negative = labels.count("NEGATIVE")
        neutral = labels.count("NEUTRAL")

        return {
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "positive_rate": round(100 * positive / total, 1),
            "negative_rate": round(100 * negative / total, 1),
            "neutral_rate": round(100 * neutral / total, 1),
        }

    def _summarize_dict(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """Summarize category/cause results."""
        return {key: round(float(np.mean(values)), 3) for key, values in data.items()}

    def _detect_priority_issues(
        self,
        sentiment_results: List[Dict[str, Any]],
        category_results: Dict[str, List[float]],
        cause_results: Dict[str, List[float]],
        texts: List[str],
    ) -> List[PriorityIssue]:
        """
        Detect priority issues using multi-factor scoring.

        - Frequency: 40%
        - Negative sentiment: 30%
        - Confidence: 20%
        - Cause correlation: 10%
        """
        if not category_results:
            return []

        issues = []
        text_count = max(1, len(texts))
        has_negative = any(r.get("label") == "NEGATIVE" for r in sentiment_results)

        for cat_name, cat_scores in category_results.items():
            affected_entries = len(cat_scores)
            avg_confidence = float(np.mean(cat_scores))
            frequency_factor = min(affected_entries / text_count, 1.0) * 0.40
            sentiment_factor = 0.30 if has_negative else 0.0
            confidence_factor = avg_confidence * 0.20
            cause_factor = (
                0.10
                if cause_results and cat_name in {"wait times", "staff behavior"}
                else 0.0
            )

            priority_score = (
                frequency_factor
                + sentiment_factor
                + confidence_factor
                + cause_factor
            )

            if priority_score > 0.60 and affected_entries > text_count * 0.30:
                severity = Severity.CRITICAL
            elif priority_score > 0.50 or affected_entries > text_count * 0.20:
                severity = Severity.HIGH
            elif affected_entries > text_count * 0.10:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            if affected_entries >= max(1, text_count * 0.10):
                issues.append(
                    PriorityIssue(
                        rank=len(issues) + 1,
                        issue_type=cat_name,
                        severity=severity,
                        frequency=len(cat_scores),
                        affected_entries=affected_entries,
                        confidence=avg_confidence,
                        avg_sentiment_score=priority_score,
                        recommended_action=self.action_map.get(
                            cat_name, "Monitor this issue"
                        ),
                    )
                )

        severity_rank = {
            "critical": 3,
            "high": 2,
            "medium": 1,
            "low": 0,
        }
        issues.sort(
            key=lambda issue: (
                -severity_rank[issue.severity.value],
                -issue.affected_entries,
            )
        )

        for i, issue in enumerate(issues, 1):
            issue.rank = i

        return issues[:10]

    def _generate_recommendations(
        self, priority_issues: List[PriorityIssue], sentiment_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if sentiment_summary["negative_rate"] > 40:
            recommendations.append(
                "CRITICAL: Negative sentiment >40%. Immediate action required on top 2 issues."
            )

        for issue in priority_issues[:3]:
            recommendations.append(issue.recommended_action)

        if not recommendations:
            recommendations.append(
                "Overall positive feedback. Continue current practices."
            )

        return recommendations

    def generate_report(self) -> str:
        """Generate formatted text report."""
        if not self.last_insight:
            return "No analysis run yet."

        insight = self.last_insight
        report = f"""
CX-INTEL AGENT ANALYSIS REPORT
==============================

ANALYSIS SUMMARY
Analysis Type:      {insight.analysis_type.value.upper()}
Total Entries:      {insight.total_entries}
Timestamp:          {insight.timestamp}

SENTIMENT DISTRIBUTION
Positive:           {insight.sentiment_summary['positive_count']} ({insight.sentiment_summary['positive_rate']}%)
Negative:           {insight.sentiment_summary['negative_count']} ({insight.sentiment_summary['negative_rate']}%)
Neutral:            {insight.sentiment_summary['neutral_count']} ({insight.sentiment_summary['neutral_rate']}%)

PRIORITY ISSUES
"""

        for issue in insight.priority_issues[:5]:
            report += f"""
{issue.rank}. {issue.issue_type.upper()} [{issue.severity.value.upper()}]
   Affected Entries: {issue.affected_entries}
   Confidence:       {issue.confidence:.1%}
   Action:           {issue.recommended_action}
"""

        report += "\nMANAGEMENT RECOMMENDATIONS\n"
        for i, rec in enumerate(insight.recommendations, 1):
            report += f"{i}. {rec}\n"

        report += f"""
Model: RoBERTa (Sentiment) + DeBERTa (Classification)
Generated: {insight.timestamp}
"""

        return report

    @staticmethod
    def _normalize_sentiment_label(label: Any) -> str:
        """Normalize common Hugging Face sentiment labels."""
        normalized = str(label).strip().upper()
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE",
            "NEGATIVE": "NEGATIVE",
            "NEUTRAL": "NEUTRAL",
            "POSITIVE": "POSITIVE",
        }
        return label_map.get(normalized, normalized)
