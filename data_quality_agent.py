"""
Data Quality Agent for Agricultural Time Series

Assesses data quality of yield curves and flags problematic plots
before they're used for training. Prevents "garbage in, garbage out".
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityIssue:
    """Represents a data quality problem"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'missing_data', 'outlier', 'anomaly', 'suspicious_pattern', 'insufficient_data'
    description: str
    affected_weeks: Optional[List[int]] = None


@dataclass
class QualityReport:
    """Complete quality assessment for a plot"""
    plot_id: int
    overall_score: float  # 0-100, higher is better
    is_usable: bool  # Should we use this for training?
    issues: List[QualityIssue]
    statistics: Dict[str, float]

    def summary(self) -> str:
        """Get human-readable summary"""
        if self.is_usable:
            status = "✓ USABLE"
        else:
            status = "✗ UNUSABLE"

        critical = sum(1 for i in self.issues if i.severity == 'critical')
        warnings = sum(1 for i in self.issues if i.severity == 'warning')

        return f"Plot {self.plot_id}: {status} (Score: {self.overall_score:.1f}/100, {critical} critical, {warnings} warnings)"


class DataQualityAgent:
    """
    Agent that assesses data quality of agricultural yield curves
    Uses both statistical analysis and LLM reasoning
    """

    def __init__(self, llm_agent=None, strict_mode: bool = False):
        """
        Initialize quality agent

        Args:
            llm_agent: Optional LLM agent for qualitative assessment
            strict_mode: If True, reject more plots (higher quality bar)
        """
        self.llm_agent = llm_agent
        self.strict_mode = strict_mode

        # Thresholds for quality checks
        self.MIN_WEEKS = 4 if not strict_mode else 8
        self.MAX_MISSING_PCT = 0.3 if not strict_mode else 0.15
        self.MAX_ZERO_PCT = 0.2 if not strict_mode else 0.1
        self.MIN_VARIANCE = 0.01  # Yield should vary somewhat
        self.MAX_CV = 2.0  # Coefficient of variation shouldn't be crazy high
        self.OUTLIER_THRESHOLD = 4.0  # Z-score threshold
        self.MAX_JUMP_FACTOR = 5.0  # Max week-to-week change as multiple of std

    def assess_plot_quality(self,
                           plot_id: int,
                           time_series: pd.DataFrame,
                           metadata: Dict) -> QualityReport:
        """
        Comprehensive quality assessment of a single plot

        Args:
            plot_id: Plot identifier
            time_series: DataFrame with 'target' column
            metadata: Plot metadata

        Returns:
            QualityReport with score and issues
        """
        issues = []
        statistics = {}

        # Extract yield values
        if 'target' not in time_series.columns:
            return QualityReport(
                plot_id=plot_id,
                overall_score=0.0,
                is_usable=False,
                issues=[QualityIssue('critical', 'missing_data', 'No target column found')],
                statistics={}
            )

        values = time_series['target'].values
        statistics['num_weeks'] = len(values)
        statistics['num_valid'] = np.sum(~np.isnan(values))
        statistics['num_missing'] = np.sum(np.isnan(values))

        # Check 1: Sufficient data
        if len(values) < self.MIN_WEEKS:
            issues.append(QualityIssue(
                'critical',
                'insufficient_data',
                f'Only {len(values)} weeks of data (minimum: {self.MIN_WEEKS})'
            ))

        # Check 2: Missing data percentage
        missing_pct = statistics['num_missing'] / len(values)
        statistics['missing_pct'] = missing_pct
        if missing_pct > self.MAX_MISSING_PCT:
            issues.append(QualityIssue(
                'critical' if missing_pct > 0.5 else 'warning',
                'missing_data',
                f'{missing_pct*100:.1f}% of data is missing'
            ))

        # Work with valid (non-NaN) values for remaining checks
        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            return QualityReport(
                plot_id=plot_id,
                overall_score=0.0,
                is_usable=False,
                issues=issues + [QualityIssue('critical', 'missing_data', 'All values are NaN')],
                statistics=statistics
            )

        # Check 3: Negative or zero values
        num_negative = np.sum(valid_values < 0)
        num_zero = np.sum(valid_values == 0)
        statistics['num_negative'] = num_negative
        statistics['num_zero'] = num_zero

        if num_negative > 0:
            issues.append(QualityIssue(
                'critical',
                'anomaly',
                f'{num_negative} negative yield values (impossible)'
            ))

        zero_pct = num_zero / len(valid_values)
        if zero_pct > self.MAX_ZERO_PCT:
            issues.append(QualityIssue(
                'warning',
                'suspicious_pattern',
                f'{zero_pct*100:.1f}% of values are exactly zero'
            ))

        # Check 4: Statistical properties
        statistics['mean'] = float(np.mean(valid_values))
        statistics['std'] = float(np.std(valid_values))
        statistics['min'] = float(np.min(valid_values))
        statistics['max'] = float(np.max(valid_values))
        statistics['cv'] = statistics['std'] / (statistics['mean'] + 1e-8)

        # Check for zero variance (constant values)
        if statistics['std'] < self.MIN_VARIANCE:
            issues.append(QualityIssue(
                'warning',
                'suspicious_pattern',
                f'Almost no variation in yield (std={statistics["std"]:.4f})'
            ))

        # Check for excessive variation
        if statistics['cv'] > self.MAX_CV:
            issues.append(QualityIssue(
                'warning',
                'outlier',
                f'Very high coefficient of variation ({statistics["cv"]:.2f})'
            ))

        # Check 5: Outliers (Z-score method)
        z_scores = np.abs((valid_values - statistics['mean']) / (statistics['std'] + 1e-8))
        outlier_mask = z_scores > self.OUTLIER_THRESHOLD
        num_outliers = np.sum(outlier_mask)
        statistics['num_outliers'] = num_outliers

        if num_outliers > 0:
            outlier_weeks = np.where(outlier_mask)[0].tolist()
            severity = 'critical' if num_outliers > len(valid_values) * 0.2 else 'warning'
            issues.append(QualityIssue(
                severity,
                'outlier',
                f'{num_outliers} extreme outliers detected (>4 std devs)',
                affected_weeks=outlier_weeks
            ))

        # Check 6: Unrealistic week-to-week jumps
        diffs = np.diff(valid_values)
        if len(diffs) > 0:
            max_jump = np.max(np.abs(diffs))
            statistics['max_jump'] = float(max_jump)
            statistics['max_jump_relative'] = max_jump / (statistics['std'] + 1e-8)

            if statistics['max_jump_relative'] > self.MAX_JUMP_FACTOR:
                issues.append(QualityIssue(
                    'warning',
                    'anomaly',
                    f'Unrealistic jump of {max_jump:.2f} ({statistics["max_jump_relative"]:.1f}x std dev)'
                ))

        # Check 7: Monotonic sequences (suspiciously smooth)
        if len(valid_values) >= 5:
            is_increasing = np.all(np.diff(valid_values) >= 0)
            is_decreasing = np.all(np.diff(valid_values) <= 0)

            if is_increasing or is_decreasing:
                issues.append(QualityIssue(
                    'info',
                    'suspicious_pattern',
                    'Perfectly monotonic sequence (unusually smooth)'
                ))

        # Check 8: Duplicate values (suspiciously many same values)
        unique_values = len(np.unique(valid_values))
        duplicate_pct = 1.0 - (unique_values / len(valid_values))
        statistics['duplicate_pct'] = duplicate_pct

        if duplicate_pct > 0.5 and unique_values < 5:
            issues.append(QualityIssue(
                'warning',
                'suspicious_pattern',
                f'Only {unique_values} unique values in {len(valid_values)} weeks'
            ))

        # Calculate overall quality score
        score = self._calculate_quality_score(statistics, issues)

        # Determine if plot is usable
        critical_issues = [i for i in issues if i.severity == 'critical']
        is_usable = len(critical_issues) == 0 and score >= (70 if self.strict_mode else 50)

        return QualityReport(
            plot_id=plot_id,
            overall_score=score,
            is_usable=is_usable,
            issues=issues,
            statistics=statistics
        )

    def _calculate_quality_score(self, statistics: Dict, issues: List[QualityIssue]) -> float:
        """
        Calculate overall quality score (0-100)

        Args:
            statistics: Statistical measures
            issues: List of detected issues

        Returns:
            Quality score from 0 to 100
        """
        score = 100.0

        # Deduct points for issues
        for issue in issues:
            if issue.severity == 'critical':
                score -= 30
            elif issue.severity == 'warning':
                score -= 10
            elif issue.severity == 'info':
                score -= 5

        # Deduct points for statistical issues
        if statistics.get('missing_pct', 0) > 0:
            score -= statistics['missing_pct'] * 20

        if statistics.get('cv', 0) > 1.0:
            score -= (statistics['cv'] - 1.0) * 10

        # Bonus points for good characteristics
        if statistics.get('num_weeks', 0) >= 12:
            score += 5

        if 0.1 < statistics.get('cv', 0) < 0.5:  # Reasonable variation
            score += 5

        return max(0.0, min(100.0, score))

    def assess_multiple_plots(self,
                             plots: List[Tuple[int, pd.DataFrame, Dict]],
                             verbose: bool = True) -> List[QualityReport]:
        """
        Assess quality of multiple plots

        Args:
            plots: List of (plot_id, time_series, metadata) tuples
            verbose: Print progress

        Returns:
            List of QualityReports
        """
        reports = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"DATA QUALITY ASSESSMENT")
            print(f"{'='*60}")
            print(f"Assessing {len(plots)} plots...")

        for plot_id, time_series, metadata in plots:
            report = self.assess_plot_quality(plot_id, time_series, metadata)
            reports.append(report)

            if verbose and not report.is_usable:
                print(f"  ⚠️  {report.summary()}")

        if verbose:
            usable = sum(1 for r in reports if r.is_usable)
            unusable = len(reports) - usable
            avg_score = np.mean([r.overall_score for r in reports])

            print(f"\n{'='*60}")
            print(f"QUALITY SUMMARY")
            print(f"{'='*60}")
            print(f"✓ Usable plots: {usable}/{len(reports)} ({usable/len(reports)*100:.1f}%)")
            print(f"✗ Unusable plots: {unusable}")
            print(f"Average quality score: {avg_score:.1f}/100")

            if unusable > 0:
                print(f"\n⚠️  {unusable} plots flagged as unusable and will be excluded from training")

        return reports

    def filter_by_quality(self,
                         plots_with_reports: List[Tuple[int, pd.DataFrame, Dict, QualityReport]],
                         min_score: Optional[float] = None) -> List[Tuple[int, pd.DataFrame, Dict]]:
        """
        Filter plots by quality, keeping only usable ones

        Args:
            plots_with_reports: List of (plot_id, time_series, metadata, report)
            min_score: Optional minimum score threshold

        Returns:
            Filtered list of (plot_id, time_series, metadata)
        """
        if min_score is None:
            min_score = 70 if self.strict_mode else 50

        filtered = []
        for plot_id, ts, meta, report in plots_with_reports:
            if report.is_usable and report.overall_score >= min_score:
                filtered.append((plot_id, ts, meta))

        return filtered

    def get_llm_assessment(self,
                          plot_id: int,
                          time_series: pd.DataFrame,
                          statistics: Dict,
                          issues: List[QualityIssue]) -> str:
        """
        Ask LLM to provide qualitative assessment

        Args:
            plot_id: Plot identifier
            time_series: Time series data
            statistics: Statistical summary
            issues: Detected issues

        Returns:
            LLM's assessment as text
        """
        if not self.llm_agent:
            return "LLM not available"

        # Prepare summary for LLM
        values = time_series['target'].dropna().values

        prompt = f"""You are an agricultural data quality expert. Assess this yield data:

Plot ID: {plot_id}

Statistics:
- Weeks of data: {statistics.get('num_weeks', 'N/A')}
- Mean yield: {statistics.get('mean', 'N/A'):.2f}
- Std dev: {statistics.get('std', 'N/A'):.2f}
- Min/Max: {statistics.get('min', 'N/A'):.2f} / {statistics.get('max', 'N/A'):.2f}
- Missing data: {statistics.get('missing_pct', 0)*100:.1f}%
- Outliers: {statistics.get('num_outliers', 0)}

Detected Issues:
"""
        for issue in issues:
            prompt += f"- [{issue.severity.upper()}] {issue.description}\n"

        if not issues:
            prompt += "- No issues detected\n"

        prompt += f"""
Sample values (first 10 weeks): {values[:10].tolist()}

Question: Is this data suitable for training a yield prediction model?
Consider: data completeness, realistic values, measurement errors, agricultural plausibility.

Answer in 2-3 sentences, focusing on PRACTICAL implications for model training."""

        return self.llm_agent.query(prompt, stream=False)

    def generate_quality_report_csv(self,
                                   reports: List[QualityReport],
                                   output_file: str):
        """
        Save quality reports to CSV

        Args:
            reports: List of quality reports
            output_file: Path to save CSV
        """
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'plot_id', 'quality_score', 'is_usable', 'num_issues',
                'num_critical', 'num_warnings',
                'num_weeks', 'missing_pct', 'mean_yield', 'std_yield', 'cv',
                'num_outliers', 'issues_summary'
            ])

            # Data
            for report in reports:
                critical = sum(1 for i in report.issues if i.severity == 'critical')
                warnings = sum(1 for i in report.issues if i.severity == 'warning')
                issues_summary = '; '.join([f"{i.severity}: {i.description}" for i in report.issues[:3]])

                writer.writerow([
                    report.plot_id,
                    f"{report.overall_score:.1f}",
                    report.is_usable,
                    len(report.issues),
                    critical,
                    warnings,
                    report.statistics.get('num_weeks', ''),
                    f"{report.statistics.get('missing_pct', 0)*100:.1f}%",
                    f"{report.statistics.get('mean', 0):.2f}",
                    f"{report.statistics.get('std', 0):.2f}",
                    f"{report.statistics.get('cv', 0):.3f}",
                    report.statistics.get('num_outliers', 0),
                    issues_summary
                ])

        print(f"✓ Quality report saved to: {output_file}")


if __name__ == "__main__":
    # Test the agent with synthetic data
    import pandas as pd

    print("Testing Data Quality Agent...")

    agent = DataQualityAgent(strict_mode=False)

    # Create test data
    dates = pd.date_range('2023-01-01', periods=20, freq='W')

    # Good quality data
    good_data = pd.DataFrame({
        'target': np.random.normal(10, 2, 20)
    }, index=dates)

    # Bad quality data (with issues)
    bad_data = pd.DataFrame({
        'target': [0, 0, 100, -5, np.nan, np.nan, 10, 10, 10, 10,
                   10, 10, 10, 10, 500, 8, 9, np.nan, 12, 11]
    }, index=dates)

    # Assess both
    good_report = agent.assess_plot_quality(1, good_data, {})
    bad_report = agent.assess_plot_quality(2, bad_data, {})

    print("\n" + "="*60)
    print("GOOD DATA:")
    print(good_report.summary())
    for issue in good_report.issues:
        print(f"  - {issue.description}")

    print("\n" + "="*60)
    print("BAD DATA:")
    print(bad_report.summary())
    for issue in bad_report.issues:
        print(f"  - [{issue.severity.upper()}] {issue.description}")

    print("\n✓ Data Quality Agent test complete!")
