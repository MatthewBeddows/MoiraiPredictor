"""
Weak Plot Identifier Agent

Autonomous agent that finds training plots that HURT predictions
despite having high similarity scores.

This is a true AI-powered agent with:
- Memory: Tracks all experiments and learns from results
- Autonomy: Runs experiments until goal achieved
- Self-critique: Uses LLM to reason about why plots are harmful
- Learning: Adjusts search strategy based on findings
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from .base_agent import BaseAgent, Critique
import time


class WeakPlotIdentifierAgent(BaseAgent):
    """
    Identifies "weak plots" - training data that increases MAE despite high similarity

    How it works:
    1. Establish baseline MAE with all training plots
    2. Systematically test removing each candidate plot
    3. Use LLM to reason about why certain plots hurt performance
    4. Learn patterns about harmful plots
    5. Return cleaned training set with weak plots removed
    """

    def __init__(self,
                 experiment_runner,  # Function that trains & evaluates model
                 llm_agent=None,
                 verbose: bool = True,
                 max_iterations: int = 30,
                 min_improvement_threshold: float = 0.005):  # 0.5% MAE improvement
        """
        Initialize Weak Plot Identifier

        Args:
            experiment_runner: Function(training_ids, test_ids) -> mae
            llm_agent: LLM for reasoning about weak plots
            verbose: Print agent reasoning
            max_iterations: Max plots to test
            min_improvement_threshold: Minimum MAE improvement to consider plot "weak"
        """
        # Store these BEFORE calling super().__init__ (which calls _register_tools)
        self.experiment_runner = experiment_runner
        self.min_improvement = min_improvement_threshold

        super().__init__(
            role="Weak Plot Detective",
            goal="Identify training plots that increase MAE despite high similarity",
            backstory="Expert at finding harmful data through systematic experimentation and causal reasoning",
            llm_agent=llm_agent,
            verbose=verbose,
            max_iterations=max_iterations
        )

        # Agent state
        self.baseline_mae = None
        self.current_mae = None
        self.weak_plots = set()
        self.tested_plots = set()
        self.candidate_scores = {}  # plot_id -> suspicion_score

    def _register_tools(self):
        """Register tools for this agent"""
        self.register_tool(
            "run_experiment",
            "Train model and evaluate MAE with given training plots",
            self.experiment_runner,
            {"training_ids": "list", "test_ids": "list"}
        )

        self.register_tool(
            "compute_suspicion_score",
            "Compute how suspicious a plot is based on characteristics",
            self._compute_suspicion_score,
            {"plot_metadata": "dict"}
        )

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main autonomous loop - find all weak plots

        Args:
            task: {
                'training_ids': List of training plot IDs,
                'test_ids': List of test plot IDs,
                'metadata_lookup': Dict mapping plot_id -> metadata,
                'initial_mae': Optional baseline MAE (will compute if not provided)
            }

        Returns:
            {
                'weak_plots': List of identified weak plot IDs,
                'clean_training_ids': Training IDs with weak plots removed,
                'baseline_mae': Initial MAE,
                'final_mae': MAE after removing weak plots,
                'improvement': Absolute MAE improvement,
                'improvement_pct': Percentage improvement,
                'experiment_count': Number of experiments run,
                'reasoning': LLM's reasoning about weak plots
            }
        """
        self.log(f"\n{'='*60}", "info")
        self.log(f"WEAK PLOT IDENTIFIER AGENT", "info")
        self.log(f"{'='*60}", "info")
        self.log(self.get_state_summary(), "info")

        # Extract task data
        training_ids = task['training_ids']
        test_ids = task['test_ids']
        metadata_lookup = task['metadata_lookup']
        initial_mae = task.get('initial_mae', None)

        self.log(f"Training plots: {len(training_ids)}", "info")
        self.log(f"Test plots: {len(test_ids)}", "info")

        # Phase 1: Establish baseline
        self.log("\n📊 Phase 1: Establishing baseline MAE...", "action")

        if initial_mae is not None:
            self.baseline_mae = initial_mae
            self.log(f"Using provided baseline: {self.baseline_mae:.4f}", "info")
        else:
            self.baseline_mae = self.use_tool(
                "run_experiment",
                training_ids=training_ids,
                test_ids=test_ids
            )
            self.log(f"Baseline MAE: {self.baseline_mae:.4f}", "success")

        self.current_mae = self.baseline_mae
        self.learn('initial_training_size', len(training_ids))
        self.learn('baseline_mae', self.baseline_mae)

        # Phase 2: Identify suspicious candidates
        self.log("\n🔍 Phase 2: Identifying suspicious candidates...", "action")

        candidates = self._identify_candidates(training_ids, metadata_lookup)
        self.log(f"Found {len(candidates)} suspicious plots to test", "success")

        if self.verbose and len(candidates) > 0:
            self.log("Top 5 most suspicious:", "debug")
            for i, (pid, score) in enumerate(candidates[:5], 1):
                meta = metadata_lookup[pid]
                self.log(f"  {i}. Plot {pid} (suspicion: {score:.3f}) - "
                        f"farm={meta.get('farm')}, year={meta.get('year')}, "
                        f"pattern={meta.get('growth_pattern')}", "debug")

        # Phase 3: Systematic testing
        self.log("\n🧪 Phase 3: Testing candidates systematically...", "action")

        current_training = training_ids.copy()

        for iteration, (plot_id, suspicion) in enumerate(candidates, 1):
            if self.check_convergence():
                break

            self.iteration_count = iteration

            # Skip if already tested
            if plot_id in self.tested_plots:
                continue

            self.log(f"\n--- Iteration {iteration}/{len(candidates)} ---", "thinking")
            self.log(f"Testing plot {plot_id} (suspicion: {suspicion:.3f})", "action")

            # Run experiment WITHOUT this plot
            training_without = [p for p in current_training if p != plot_id]

            mae_without = self.use_tool(
                "run_experiment",
                training_ids=training_without,
                test_ids=test_ids
            )

            improvement = self.current_mae - mae_without
            improvement_pct = (improvement / self.current_mae) * 100

            self.log(f"MAE with plot: {self.current_mae:.4f}", "debug")
            self.log(f"MAE without: {mae_without:.4f}", "debug")
            self.log(f"Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)", "debug")

            # Self-critique: Is this a weak plot?
            critique = self._critique_experiment(
                plot_id=plot_id,
                metadata=metadata_lookup[plot_id],
                mae_with=self.current_mae,
                mae_without=mae_without,
                improvement=improvement,
                suspicion=suspicion
            )

            # Remember this experiment
            self.remember(
                action="test_plot_removal",
                params={'plot_id': plot_id, 'suspicion': suspicion},
                result={
                    'mae_with': self.current_mae,
                    'mae_without': mae_without,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                },
                critique=critique
            )

            self.tested_plots.add(plot_id)

            # Decision: Keep or remove this plot?
            if critique.is_success and improvement > self.min_improvement:
                self.log(f"✓ Weak plot identified: {plot_id} (MAE improved by {improvement:.4f})", "success")
                self.weak_plots.add(plot_id)
                current_training = training_without
                self.current_mae = mae_without

                # Learn from this finding
                self._learn_from_weak_plot(plot_id, metadata_lookup[plot_id], critique)
            else:
                self.log(f"✗ Plot {plot_id} is NOT weak (keeping it)", "info")

        # Phase 4: Final summary
        self.log("\n📋 Phase 4: Generating summary...", "action")

        total_improvement = self.baseline_mae - self.current_mae
        improvement_pct = (total_improvement / self.baseline_mae) * 100

        summary = self._generate_summary(
            training_ids=training_ids,
            metadata_lookup=metadata_lookup,
            total_improvement=total_improvement,
            improvement_pct=improvement_pct
        )

        result = {
            'weak_plots': list(self.weak_plots),
            'clean_training_ids': current_training,
            'baseline_mae': self.baseline_mae,
            'final_mae': self.current_mae,
            'improvement': total_improvement,
            'improvement_pct': improvement_pct,
            'experiment_count': self.iteration_count,
            'reasoning': summary,
            'patterns_learned': dict(self.long_term_memory)
        }

        self.log("\n" + "="*60, "info")
        self.log("WEAK PLOT IDENTIFICATION COMPLETE", "success")
        self.log("="*60, "info")
        self.log(f"Weak plots found: {len(self.weak_plots)}", "success")
        self.log(f"MAE improvement: {total_improvement:.4f} ({improvement_pct:+.2f}%)", "success")
        self.log(f"Final MAE: {self.current_mae:.4f} (was {self.baseline_mae:.4f})", "success")

        return result

    def _identify_candidates(self,
                            training_ids: List[int],
                            metadata_lookup: Dict) -> List[Tuple[int, float]]:
        """
        Identify suspicious plots to test

        Uses heuristics + LLM reasoning to prioritize which plots to test first

        Returns:
            List of (plot_id, suspicion_score) sorted by suspicion (high to low)
        """
        candidates = []

        for plot_id in training_ids:
            metadata = metadata_lookup[plot_id]

            # Compute suspicion score using tool
            suspicion = self.use_tool("compute_suspicion_score", plot_metadata=metadata)

            candidates.append((plot_id, suspicion))
            self.candidate_scores[plot_id] = suspicion

        # Sort by suspicion (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # If LLM available, ask for additional reasoning
        if self.llm_agent and len(candidates) > 0:
            llm_insights = self._get_llm_candidate_insights(candidates[:10], metadata_lookup)
            if self.verbose and llm_insights:
                self.log("LLM insights on candidates:", "thinking")
                self.log(llm_insights, "debug")

        return candidates

    def _compute_suspicion_score(self, plot_metadata: Dict) -> float:
        """
        Compute how suspicious/risky a plot is

        Higher score = more likely to be harmful

        Heuristics:
        - Old data (2020) more suspicious than recent (2022)
        - High volatility suspicious
        - Extreme values suspicious
        - Different growth pattern than majority suspicious
        """
        suspicion = 0.0

        # Year: older = more suspicious (2020 = +0.3, 2021 = +0.15, 2022 = +0)
        year = plot_metadata.get('year', 2022)
        if year == 2020:
            suspicion += 0.3
        elif year == 2021:
            suspicion += 0.15

        # Volatility: high volatility = suspicious
        volatility = plot_metadata.get('volatility', 0)
        if volatility > 0.5:
            suspicion += 0.2
        elif volatility > 0.3:
            suspicion += 0.1

        # CV (coefficient of variation): very high or very low suspicious
        cv = plot_metadata.get('cv_yield', 0)
        if cv > 1.0 or cv < 0.05:
            suspicion += 0.2

        # Growth pattern: if we've learned certain patterns are bad
        pattern = plot_metadata.get('growth_pattern', '')
        weak_patterns = self.retrieve_learning('weak_patterns', set())
        if pattern in weak_patterns:
            suspicion += 0.25

        # Farm: if we've learned certain farms are problematic
        farm = plot_metadata.get('farm', 0)
        weak_farms = self.retrieve_learning('weak_farms', set())
        if farm in weak_farms:
            suspicion += 0.3

        return min(suspicion, 1.0)  # Cap at 1.0

    def _critique_experiment(self,
                            plot_id: int,
                            metadata: Dict,
                            mae_with: float,
                            mae_without: float,
                            improvement: float,
                            suspicion: float) -> Critique:
        """
        Use LLM to reason about whether plot is weak
        """
        # Build critique prompt
        prompt = f"""You are an expert agricultural data scientist analyzing training data quality.

Experiment Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plot ID: {plot_id}
MAE WITH this plot: {mae_with:.4f}
MAE WITHOUT this plot: {mae_without:.4f}
Improvement from removal: {improvement:.4f} ({(improvement/mae_with)*100:+.2f}%)

Plot Characteristics:
- Farm: {metadata.get('farm', 'unknown')}
- Year: {metadata.get('year', 'unknown')}
- Growth pattern: {metadata.get('growth_pattern', 'unknown')}
- Mean yield: {metadata.get('mean_yield', 0):.2f}
- Volatility: {metadata.get('volatility', 0):.3f}
- CV: {metadata.get('cv_yield', 0):.3f}
- Suspicion score: {suspicion:.3f}

Question: Should we REMOVE this plot from training data?

Consider:
1. Is the MAE improvement significant (>{self.min_improvement})?
2. Why might this plot hurt predictions despite similarity?
3. What patterns do you notice (farm, year, growth pattern, volatility)?
4. Could this be random noise or a real signal?

Provide your reasoning and a clear YES/NO decision.
Format:
Decision: YES or NO
Confidence: 0-100%
Reasoning: [your analysis]
Patterns: [any patterns you notice]
"""

        if self.llm_agent:
            response = self.llm_agent.query(prompt, stream=False)

            # Parse response
            is_weak = improvement > self.min_improvement and \
                     any(word in response.lower() for word in ['yes', 'remove', 'weak', 'harmful'])

            # Extract confidence if present
            import re
            confidence_match = re.search(r'confidence[:\s]+([0-9]+)', response.lower())
            confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.5

            return Critique(
                is_success=is_weak,
                score=confidence,
                reasoning=response,
                suggestions=[],
                next_action="remove_plot" if is_weak else "keep_plot"
            )
        else:
            # Fallback: simple heuristic
            is_weak = improvement > self.min_improvement
            return Critique(
                is_success=is_weak,
                score=improvement / mae_with,
                reasoning=f"Improvement of {improvement:.4f} {'exceeds' if is_weak else 'below'} threshold {self.min_improvement}",
                suggestions=[],
                next_action="remove_plot" if is_weak else "keep_plot"
            )

    def _learn_from_weak_plot(self, plot_id: int, metadata: Dict, critique: Critique):
        """
        Extract patterns from weak plots to improve future detection
        """
        # Learn about weak farms
        farm = metadata.get('farm')
        if farm:
            weak_farms = self.retrieve_learning('weak_farms', set())
            weak_farms.add(farm)
            self.learn('weak_farms', weak_farms)

        # Learn about weak patterns
        pattern = metadata.get('growth_pattern')
        if pattern:
            weak_patterns = self.retrieve_learning('weak_patterns', set())
            weak_patterns.add(pattern)
            self.learn('weak_patterns', weak_patterns)

        # Learn about weak years
        year = metadata.get('year')
        if year:
            weak_years = self.retrieve_learning('weak_years', set())
            weak_years.add(year)
            self.learn('weak_years', weak_years)

    def _get_llm_candidate_insights(self,
                                    top_candidates: List[Tuple[int, float]],
                                    metadata_lookup: Dict) -> str:
        """Ask LLM for insights on top candidates"""
        if not self.llm_agent or len(top_candidates) == 0:
            return ""

        candidate_summary = "\n".join([
            f"Plot {pid}: suspicion={score:.3f}, "
            f"farm={metadata_lookup[pid].get('farm')}, "
            f"year={metadata_lookup[pid].get('year')}, "
            f"pattern={metadata_lookup[pid].get('growth_pattern')}"
            for pid, score in top_candidates[:10]
        ])

        prompt = f"""You are analyzing agricultural training data to identify potentially harmful plots.

Top 10 most suspicious plots:
{candidate_summary}

Question: Based on these characteristics, what patterns suggest these plots might hurt model performance?

Consider:
- Are certain farms or years over-represented?
- Do certain growth patterns seem problematic?
- What makes these plots "different" from typical data?

Provide 2-3 key insights in bullet points.
"""

        return self.llm_agent.query(prompt, stream=False)

    def _generate_summary(self,
                         training_ids: List[int],
                         metadata_lookup: Dict,
                         total_improvement: float,
                         improvement_pct: float) -> str:
        """Generate final summary with LLM reasoning"""

        if not self.llm_agent or len(self.weak_plots) == 0:
            return f"Removed {len(self.weak_plots)} weak plots. MAE improved by {total_improvement:.4f} ({improvement_pct:+.2f}%)."

        # Build summary of weak plots
        weak_summary = "\n".join([
            f"Plot {pid}: farm={metadata_lookup[pid].get('farm')}, "
            f"year={metadata_lookup[pid].get('year')}, "
            f"pattern={metadata_lookup[pid].get('growth_pattern')}, "
            f"suspicion={self.candidate_scores.get(pid, 0):.3f}"
            for pid in self.weak_plots
        ])

        prompt = f"""You are summarizing the results of training data quality analysis.

Findings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Initial training plots: {len(training_ids)}
- Weak plots identified: {len(self.weak_plots)}
- Baseline MAE: {self.baseline_mae:.4f}
- Final MAE: {self.current_mae:.4f}
- Total improvement: {total_improvement:.4f} ({improvement_pct:+.2f}%)
- Experiments run: {self.iteration_count}

Weak Plots Removed:
{weak_summary}

Learned Patterns:
- Weak farms: {list(self.retrieve_learning('weak_farms', set()))}
- Weak years: {list(self.retrieve_learning('weak_years', set()))}
- Weak patterns: {list(self.retrieve_learning('weak_patterns', set()))}

Task: Provide a concise 3-4 sentence summary explaining:
1. What patterns made these plots harmful
2. Why removing them improved MAE
3. Recommendations for future data collection

Write for a technical audience (data scientists).
"""

        return self.llm_agent.query(prompt, stream=False)

    def _build_critique_prompt(self, result: Dict, context: Dict) -> str:
        """Build critique prompt (required by base class)"""
        return f"Evaluate this weak plot identification result: {result}"

    def check_convergence(self) -> bool:
        """
        Check if we should stop searching

        Stop if:
        - Hit max iterations
        - No improvement in last 5 attempts
        - Found enough weak plots (>20% of training data seems excessive)
        """
        # Standard checks from base class
        if super().check_convergence():
            return True

        # Stop if we've removed too many plots (>20% seems excessive)
        initial_size = self.retrieve_learning('initial_training_size', 100)
        if len(self.weak_plots) > initial_size * 0.2:
            self.log("Removed >20% of training data, stopping to avoid over-filtering", "warning")
            return True

        return False


# Test/example usage
if __name__ == "__main__":
    print("Weak Plot Identifier Agent - Example Usage")
    print("=" * 60)

    # Mock experiment runner for testing
    def mock_experiment(training_ids, test_ids):
        """Simulate training and return mock MAE"""
        # Simulate: plot 5 and plot 12 are "weak" (removing improves MAE)
        weak_plots = {5, 12}
        mae = 0.500  # Base MAE

        # Removing weak plots improves MAE
        for weak in weak_plots:
            if weak not in training_ids:
                mae -= 0.015  # 1.5% improvement per weak plot removed

        return mae

    # Mock metadata
    metadata = {
        i: {
            'farm': (i % 3) + 1,
            'year': 2020 + (i % 3),
            'growth_pattern': ['peak_middle', 'increasing', 'decreasing'][i % 3],
            'mean_yield': 10.0 + (i % 5),
            'volatility': 0.2 + (i % 10) * 0.05,
            'cv_yield': 0.15 + (i % 8) * 0.05
        }
        for i in range(1, 21)
    }

    # Create agent (without LLM for testing)
    agent = WeakPlotIdentifierAgent(
        experiment_runner=mock_experiment,
        llm_agent=None,
        verbose=True,
        max_iterations=20,
        min_improvement_threshold=0.01
    )

    # Run agent
    result = agent.solve({
        'training_ids': list(range(1, 21)),
        'test_ids': list(range(100, 110)),
        'metadata_lookup': metadata
    })

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Weak plots: {result['weak_plots']}")
    print(f"Improvement: {result['improvement']:.4f} ({result['improvement_pct']:.2f}%)")
    print(f"Experiments run: {result['experiment_count']}")
    print("\n✓ Test complete!")
