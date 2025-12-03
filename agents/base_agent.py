"""
Base Agent Class for AI-Powered Agents

Provides core agent capabilities:
- Memory (short-term and long-term)
- Tool management
- Self-critique loop
- Goal-oriented planning
- Learning from experience
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from abc import ABC, abstractmethod


@dataclass
class Memory:
    """Memory entry for agent"""
    timestamp: datetime
    action: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    critique: Optional[str] = None
    success: bool = False


@dataclass
class Critique:
    """Self-critique result"""
    is_success: bool
    score: float  # 0-1, higher is better
    reasoning: str
    suggestions: List[str] = field(default_factory=list)
    next_action: Optional[str] = None


@dataclass
class Tool:
    """Tool that agent can use"""
    name: str
    description: str
    function: Callable
    params_schema: Dict[str, Any]


class BaseAgent(ABC):
    """
    Base class for all AI-powered agents

    Key features:
    - Memory: Tracks all actions and results
    - Tools: Can discover and use tools
    - Autonomy: Runs in a loop until goal achieved
    - Self-critique: Evaluates own work
    - Learning: Adjusts strategy based on experience
    """

    def __init__(self,
                 role: str,
                 goal: str,
                 backstory: str = "",
                 llm_agent=None,
                 verbose: bool = True,
                 max_iterations: int = 50):
        """
        Initialize base agent

        Args:
            role: Agent's role/identity (e.g., "Meta-Learning Specialist")
            goal: What the agent should achieve (e.g., "Minimize MAE")
            backstory: Context about agent's expertise
            llm_agent: LLM for reasoning (optional but recommended)
            verbose: Print agent's reasoning
            max_iterations: Max iterations before giving up
        """
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm_agent = llm_agent
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Memory stores
        self.short_term_memory: List[Memory] = []  # Last N actions
        self.long_term_memory: Dict[str, Any] = {}  # Persistent learnings
        self.working_memory: Dict[str, Any] = {}  # Current task state

        # Tool registry
        self.tools: Dict[str, Tool] = {}

        # Performance tracking
        self.best_result = None
        self.best_score = float('-inf')
        self.iteration_count = 0

        # Initialize agent-specific tools
        self._register_tools()

    @abstractmethod
    def _register_tools(self):
        """
        Register tools that this agent can use
        Override in subclass
        """
        pass

    @abstractmethod
    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point - agent works autonomously to solve task
        Override in subclass

        Args:
            task: Task specification (varies by agent)

        Returns:
            Solution dict with results
        """
        pass

    def register_tool(self, name: str, description: str,
                     function: Callable, params_schema: Dict = None):
        """Register a tool that agent can use"""
        self.tools[name] = Tool(
            name=name,
            description=description,
            function=function,
            params_schema=params_schema or {}
        )

        if self.verbose:
            print(f"  🔧 Registered tool: {name}")

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool

        Args:
            tool_name: Name of tool to use
            **kwargs: Tool parameters

        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")

        tool = self.tools[tool_name]

        if self.verbose:
            print(f"  🔧 Using tool: {tool_name}")

        try:
            result = tool.function(**kwargs)
            return result
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Tool error: {e}")
            return None

    def remember(self, action: str, params: Dict, result: Dict,
                 critique: Optional[Critique] = None):
        """
        Store action in memory

        Args:
            action: Action taken
            params: Action parameters
            result: Action result
            critique: Optional self-critique
        """
        memory = Memory(
            timestamp=datetime.now(),
            action=action,
            params=params,
            result=result,
            critique=critique.reasoning if critique else None,
            success=critique.is_success if critique else False
        )

        self.short_term_memory.append(memory)

        # Keep only last 100 memories in short-term
        if len(self.short_term_memory) > 100:
            self.short_term_memory = self.short_term_memory[-100:]

    def recall(self, query: str = None, limit: int = 10) -> List[Memory]:
        """
        Retrieve memories

        Args:
            query: Optional filter (e.g., "successful")
            limit: Max memories to return

        Returns:
            List of relevant memories
        """
        memories = self.short_term_memory

        if query == "successful":
            memories = [m for m in memories if m.success]
        elif query == "failed":
            memories = [m for m in memories if not m.success]

        return memories[-limit:]

    def learn(self, key: str, value: Any):
        """
        Store long-term learning

        Args:
            key: Learning key (e.g., "best_k_value")
            value: Learning value
        """
        self.long_term_memory[key] = value

        if self.verbose:
            print(f"  📚 Learned: {key} = {value}")

    def retrieve_learning(self, key: str, default: Any = None) -> Any:
        """Retrieve long-term learning"""
        return self.long_term_memory.get(key, default)

    def self_critique(self, result: Dict[str, Any],
                     context: Dict[str, Any] = None) -> Critique:
        """
        Evaluate own work using LLM reasoning

        Args:
            result: Result to critique
            context: Additional context

        Returns:
            Critique with assessment and suggestions
        """
        if not self.llm_agent:
            # Fallback: Simple heuristic critique
            return self._heuristic_critique(result)

        # Use LLM for nuanced critique
        prompt = self._build_critique_prompt(result, context)

        llm_response = self.llm_agent.query(prompt, stream=False)

        # Parse LLM response into structured critique
        return self._parse_critique_response(llm_response, result)

    def _heuristic_critique(self, result: Dict[str, Any]) -> Critique:
        """
        Simple rule-based critique when LLM not available
        Override in subclass for domain-specific logic
        """
        # Default: success if result exists
        return Critique(
            is_success=result is not None,
            score=0.5,
            reasoning="No LLM available for detailed critique",
            suggestions=[],
            next_action=None
        )

    @abstractmethod
    def _build_critique_prompt(self, result: Dict, context: Dict) -> str:
        """
        Build prompt for LLM critique
        Override in subclass
        """
        pass

    def _parse_critique_response(self, llm_response: str,
                                 result: Dict) -> Critique:
        """
        Parse LLM response into structured Critique

        Args:
            llm_response: Raw LLM text
            result: Result being critiqued

        Returns:
            Structured Critique object
        """
        # Simple parsing - look for keywords
        response_lower = llm_response.lower()

        is_success = any(word in response_lower
                        for word in ['good', 'success', 'excellent', 'improvement'])

        # Extract score if present
        score = 0.5  # Default
        if 'score' in response_lower:
            # Try to extract number between 0-1 or 0-100
            import re
            match = re.search(r'score[:\s]+([0-9.]+)', response_lower)
            if match:
                val = float(match.group(1))
                score = val if val <= 1.0 else val / 100.0

        # Extract suggestions (lines starting with "- " or numbered)
        suggestions = []
        for line in llm_response.split('\n'):
            if line.strip().startswith('-') or \
               line.strip().startswith('•') or \
               any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                suggestions.append(line.strip().lstrip('-•123456789. '))

        return Critique(
            is_success=is_success,
            score=score,
            reasoning=llm_response,
            suggestions=suggestions,
            next_action=suggestions[0] if suggestions else None
        )

    def log(self, message: str, level: str = "info"):
        """
        Log agent activity

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        if not self.verbose:
            return

        icons = {
            'info': 'ℹ️',
            'debug': '🔍',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅',
            'thinking': '🤔',
            'action': '🎯'
        }

        icon = icons.get(level, 'ℹ️')
        print(f"{icon} [{self.role}] {message}")

    def check_convergence(self) -> bool:
        """
        Check if agent should stop (goal achieved or stuck)
        Override in subclass for domain-specific logic
        """
        # Default: stop after max iterations
        if self.iteration_count >= self.max_iterations:
            self.log(f"Max iterations ({self.max_iterations}) reached", "warning")
            return True

        # Check if we're stuck (last 5 attempts all failed)
        recent = self.recall(limit=5)
        if len(recent) >= 5 and all(not m.success for m in recent):
            self.log("No progress in last 5 attempts, stopping", "warning")
            return True

        return False

    def get_state_summary(self) -> str:
        """Get human-readable state summary"""
        summary = f"""
{self.role} State Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: {self.goal}
Iteration: {self.iteration_count}/{self.max_iterations}
Best Score: {self.best_score:.4f}
Memory: {len(self.short_term_memory)} actions recorded
Tools: {len(self.tools)} available
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return summary

    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'role': self.role,
            'goal': self.goal,
            'backstory': self.backstory,
            'iteration_count': self.iteration_count,
            'best_score': self.best_score,
            'best_result': self.best_result,
            'long_term_memory': self.long_term_memory,
            'short_term_memory': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'action': m.action,
                    'params': m.params,
                    'result': m.result,
                    'critique': m.critique,
                    'success': m.success
                }
                for m in self.short_term_memory
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        self.log(f"State saved to {filepath}", "success")

    def load_state(self, filepath: str):
        """Load agent state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.iteration_count = state['iteration_count']
        self.best_score = state['best_score']
        self.best_result = state['best_result']
        self.long_term_memory = state['long_term_memory']

        # Restore short-term memory
        self.short_term_memory = [
            Memory(
                timestamp=datetime.fromisoformat(m['timestamp']),
                action=m['action'],
                params=m['params'],
                result=m['result'],
                critique=m['critique'],
                success=m['success']
            )
            for m in state['short_term_memory']
        ]

        self.log(f"State loaded from {filepath}", "success")


if __name__ == "__main__":
    # Test base agent
    print("Testing BaseAgent class...")

    class TestAgent(BaseAgent):
        def _register_tools(self):
            self.register_tool(
                "test_tool",
                "A test tool",
                lambda x: x * 2,
                {"x": "number"}
            )

        def solve(self, task):
            self.log("Starting solve", "action")
            result = self.use_tool("test_tool", x=5)
            self.log(f"Result: {result}", "success")
            return {"result": result}

        def _build_critique_prompt(self, result, context):
            return f"Evaluate this result: {result}"

    agent = TestAgent(
        role="Test Agent",
        goal="Test the base class",
        verbose=True
    )

    result = agent.solve({})
    print(f"\n✓ BaseAgent test complete: {result}")
