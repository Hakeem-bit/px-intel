"""
PX-Intel Agent Core Logic
Intelligent orchestration of sentiment analysis, categorization, and causal inference
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter


# ============================================================================
# ENUMS
# ============================================================================

class AnalysisType(Enum):
    """Analysis depth selection"""
    SENTIMENT_ONLY = "sentiment_only"
    FULL_ANALYSIS = "full_analysis"
    PRIORITY_DETECTION = "priority_detection"
    CRISIS_MODE = "crisis_mode"


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PriorityIssue:
    """Represents a priority issue detected by the agent"""
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
    """Complete agent analysis output"""
    timestamp: str
    analysis_type: AnalysisType
    total_entries: int
    sentiment_summary: Dict[str, Any]
    category_summary: Dict[str, int]
    cause_summary: Dict[str, int]
    priority_issues: List[PriorityIssue]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# AGENT LOGIC
# ============================================================================

class PXIntelAgent:
    """
    Intelligent orchestration agent for hospital feedback analysis.
    
    Decision Logic:
    - <50 entries: SENTIMENT_ONLY (fast MVP)
    - 50-200 entries: FULL_ANALYSIS (balanced)
    - >200 entries: PRIORITY_DETECTION (focused)
    - >40% negative: CRISIS_MODE (urgent)
    """
    
    def __init__(self, sentiment_model, deberta_model):
        """
        Initialize agent with models
        
        Args:
            sentiment_model: RoBERTa sentiment classification pipeline
            deberta_model: DeBERTa zero-shot classification pipeline
        """
        self.sentiment_model = sentiment_model
        self.deberta_model = deberta_model
        self.last_insight: Optional[AgentInsight] = None
        
        # Mapping for recommendations
        self.action_map = {
            "wait times": "🔴 URGENT: Implement queue management system or appointment scheduling",
            "staff behavior": "🟠 PRIORITY: Conduct staff communication training and empathy workshops",
            "cleanliness": "🟡 MONITOR: Schedule facility maintenance and cleanliness audit",
            "treatment quality": "🟡 IMPORTANT: Audit clinical protocols and treatment standards",
            "costs": "🟡 FOLLOW-UP: Review billing procedures and cost transparency",
            "general": "🟢 OBSERVE: Monitor feedback trends over time"
        }
    
    def decide_analysis_type(self, df: pd.DataFrame) -> AnalysisType:
        """
        Decide what type of analysis to run based on dataset characteristics.
        
        Rules:
        - <50 entries: SENTIMENT_ONLY
        - 50-200 entries: FULL_ANALYSIS
        - >200 entries: PRIORITY_DETECTION
        - >40% negative: CRISIS_MODE (override)
        """
        n_entries = len(df)
        
        # Check if we have sentiment column (from previous analysis)
        if 'sentiment' in df.columns:
            negative_pct = (df['sentiment'] == 'NEGATIVE').sum() / len(df)
            if negative_pct > 0.40:
                return AnalysisType.CRISIS_MODE
        
        # Size-based decision
        if n_entries < 50:
            return AnalysisType.SENTIMENT_ONLY
        elif n_entries < 200:
            return AnalysisType.FULL_ANALYSIS
        else:
            return AnalysisType.PRIORITY_DETECTION
    
    def orchestrate_analysis(
        self,
        df: pd.DataFrame,
        texts: List[str]
    ) -> AgentInsight:
        """
        Run complete analysis pipeline: sentiment → categories → causes → priorities
        
        Args:
            df: DataFrame with feedback data
            texts: List of feedback texts
            
        Returns:
            AgentInsight with complete analysis
        """
        
        # Decision
        analysis_type = self.decide_analysis_type(df)
        
        # Sentiment Analysis (Phase 1)
        sentiment_results = self._sentiment_analysis(texts)
        
        # Category Analysis (Phase 2)
        if analysis_type in [AnalysisType.FULL_ANALYSIS, AnalysisType.PRIORITY_DETECTION, AnalysisType.CRISIS_MODE]:
            category_results = self._category_analysis(texts)
        else:
            category_results = {}
        
        # Cause Analysis (Phase 3)
        if analysis_type in [AnalysisType.FULL_ANALYSIS, AnalysisType.PRIORITY_DETECTION, AnalysisType.CRISIS_MODE]:
            cause_results = self._cause_analysis(texts, sentiment_results)
        else:
            cause_results = {}
        
        # Generate sentiment summary
        sentiment_summary = self._summarize_sentiment(sentiment_results)
        
        # Detect priority issues
        priority_issues = self._detect_priority_issues(
            sentiment_results,
            category_results,
            cause_results,
            texts
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(priority_issues, sentiment_summary)
        
        # Build insight
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
                "confidence_threshold": 0.5
            }
        )
        
        self.last_insight = insight
        return insight
    
    def _sentiment_analysis(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run sentiment analysis on texts"""
        results = []
        for text in texts:
            try:
                if not text or len(str(text).strip()) < 5:
                    results.append({"label": "NEUTRAL", "score": 0.5})
                    continue
                
                text_truncated = str(text)[:512]
                result = self.sentiment_model(text_truncated, truncation=True, max_length=512)
                results.append({
                    "label": result[0]["label"],
                    "score": result[0]["score"]
                })
            except Exception as e:
                results.append({"label": "NEUTRAL", "score": 0.5})
        
        return results
    
    def _category_analysis(self, texts: List[str]) -> Dict[str, List[float]]:
        """Categorize feedback into issue types"""
        hypotheses = [
            "This feedback mentions wait times or scheduling issues",
            "This feedback mentions staff behavior or communication",
            "This feedback mentions cleanliness or facility conditions",
            "This feedback mentions treatment quality or outcomes",
            "This feedback mentions costs or billing issues",
            "This feedback is general feedback or other issues"
        ]
        
        category_map = {
            0: "wait times",
            1: "staff behavior",
            2: "cleanliness",
            3: "treatment quality",
            4: "costs",
            5: "general"
        }
        
        results = {}
        
        for text in texts:
            try:
                if not text or len(str(text).strip()) < 10:
                    continue
                
                text_truncated = str(text)[:512]
                result = self.deberta_model(
                    text_truncated,
                    hypotheses,
                    multi_class=True,
                    truncation=True,
                    max_length=512
                )
                
                for idx, (label, score) in enumerate(zip(result['labels'], result['scores'])):
                    cat = category_map.get(idx, "general")
                    if cat not in results:
                        results[cat] = []
                    results[cat].append(float(score))
            except Exception as e:
                pass
        
        return results
    
    def _cause_analysis(
        self,
        texts: List[str],
        sentiment_results: List[Dict]
    ) -> Dict[str, List[float]]:
        """Analyze root causes for negative feedback"""
        cause_hypotheses = [
            "The reason for dissatisfaction is inadequate staffing",
            "The reason for dissatisfaction is poor communication",
            "The reason for dissatisfaction is facility maintenance issues",
            "The reason for dissatisfaction is long wait times",
            "The reason for dissatisfaction is lack of empathy or care"
        ]
        
        cause_map = {
            0: "inadequate staffing",
            1: "poor communication",
            2: "facility maintenance",
            3: "long wait times",
            4: "lack of empathy"
        }
        
        results = {}
        
        for text, sentiment in zip(texts, sentiment_results):
            # Only analyze negative feedback
            if sentiment.get("label") != "NEGATIVE":
                continue
            
            try:
                text_truncated = str(text)[:512]
                result = self.deberta_model(
                    text_truncated,
                    cause_hypotheses,
                    multi_class=True,
                    truncation=True,
                    max_length=512
                )
                
                for idx, (label, score) in enumerate(zip(result['labels'], result['scores'])):
                    if score >= 0.40:  # Confidence threshold
                        cause = cause_map.get(idx, "unknown")
                        if cause not in results:
                            results[cause] = []
                        results[cause].append(float(score))
            except Exception as e:
                pass
        
        return results
    
    def _summarize_sentiment(self, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Summarize sentiment analysis results"""
        if not sentiment_results:
            return {
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_rate": 0,
                "negative_rate": 0,
                "neutral_rate": 0
            }
        
        labels = [r.get("label", "NEUTRAL") for r in sentiment_results]
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
            "neutral_rate": round(100 * neutral / total, 1)
        }
    
    def _summarize_dict(self, data: Dict[str, List]) -> Dict[str, float]:
        """Summarize category/cause results"""
        return {
            key: round(np.mean(values), 3)
            for key, values in data.items()
        }
    
    def _detect_priority_issues(
        self,
        sentiment_results: List[Dict],
        category_results: Dict,
        cause_results: Dict,
        texts: List[str]
    ) -> List[PriorityIssue]:
        """
        Detect priority issues using multi-factor scoring:
        - Frequency (40%)
        - Negative sentiment (30%)
        - Confidence (20%)
        - Cause correlation (10%)
        """
        
        if not category_results:
            return []
        
        issues = []
        
        for cat_name, cat_scores in category_results.items():
            affected_entries = len(cat_scores)
            avg_confidence = np.mean(cat_scores)
            
            # Get sentiment for this category
            avg_sentiment = 0.0
            neg_sentiment_count = 0
            
            # Calculate severity based on factors
            frequency_factor = min(affected_entries / max(1, len(texts)), 1.0) * 0.40
            
            # Check if this category correlates with negative sentiment
            sentiment_factor = 0
            if any(r.get("label") == "NEGATIVE" for r in sentiment_results):
                sentiment_factor = 0.30
            
            confidence_factor = avg_confidence * 0.20
            
            # Check for cause correlation
            cause_factor = 0
            if cause_results and cat_name in ["wait times", "staff behavior"]:
                cause_factor = 0.10
            
            priority_score = frequency_factor + sentiment_factor + confidence_factor + cause_factor
            
            # Determine severity
            if priority_score > 0.60 and affected_entries > len(texts) * 0.30:
                severity = Severity.CRITICAL
            elif priority_score > 0.50 or affected_entries > len(texts) * 0.20:
                severity = Severity.HIGH
            elif affected_entries > len(texts) * 0.10:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            # Only include issues affecting >10% of entries
            if affected_entries >= max(1, len(texts) * 0.10):
                issue = PriorityIssue(
                    rank=len(issues) + 1,
                    issue_type=cat_name,
                    severity=severity,
                    frequency=len(cat_scores),
                    affected_entries=affected_entries,
                    confidence=avg_confidence,
                    avg_sentiment_score=priority_score,
                    recommended_action=self.action_map.get(cat_name, "Monitor this issue")
                )
                issues.append(issue)
        
        # Sort by rank
        issues.sort(key=lambda x: (
            -{"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}[x.severity.value],
            -x.affected_entries
        ))
        
        # Re-rank
        for i, issue in enumerate(issues, 1):
            issue.rank = i
        
        return issues[:10]  # Top 10 issues
    
    def _generate_recommendations(
        self,
        priority_issues: List[PriorityIssue],
        sentiment_summary: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Crisis alert
        if sentiment_summary["negative_rate"] > 40:
            recommendations.append(
                "🔴 CRITICAL: Negative sentiment >40%. Immediate action required on top 2 issues."
            )
        
        # Top issues
        for issue in priority_issues[:3]:
            recommendations.append(issue.recommended_action)
        
        # Summary
        if not recommendations:
            recommendations.append("✅ Overall positive feedback. Continue current practices.")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate formatted text report"""
        if not self.last_insight:
            return "No analysis run yet."
        
        insight = self.last_insight
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║         PX-INTEL AGENT ANALYSIS REPORT                          ║
╚══════════════════════════════════════════════════════════════════╝

📊 ANALYSIS SUMMARY
────────────────────────────────────────────────────────────────────
Analysis Type:      {insight.analysis_type.value.upper()}
Total Entries:      {insight.total_entries}
Timestamp:          {insight.timestamp}

📈 SENTIMENT DISTRIBUTION
────────────────────────────────────────────────────────────────────
Positive:           {insight.sentiment_summary['positive_count']} ({insight.sentiment_summary['positive_rate']}%)
Negative:           {insight.sentiment_summary['negative_count']} ({insight.sentiment_summary['negative_rate']}%)
Neutral:            {insight.sentiment_summary['neutral_count']} ({insight.sentiment_summary['neutral_rate']}%)

🎯 PRIORITY ISSUES
────────────────────────────────────────────────────────────────────
"""
        
        for issue in insight.priority_issues[:5]:
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }[issue.severity.value]
            
            report += f"""
{severity_icon} {issue.rank}. {issue.issue_type.upper()} [{issue.severity.value.upper()}]
   Affected Entries: {issue.affected_entries}
   Confidence:       {issue.confidence:.1%}
   Action:           {issue.recommended_action}
"""
        
        report += f"""
💡 MANAGEMENT RECOMMENDATIONS
────────────────────────────────────────────────────────────────────
"""
        for i, rec in enumerate(insight.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
════════════════════════════════════════════════════════════════════
Model: RoBERTa (Sentiment) + DeBERTa (Classification)
Generated: {insight.timestamp}
"""
        
        return report
