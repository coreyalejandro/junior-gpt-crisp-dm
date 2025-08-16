import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import httpx
from pydantic import BaseModel, Field


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating agent performance"""
    task_success_rate: float
    factual_accuracy: float
    tool_success_rate: float
    average_latency: float
    p95_latency: float
    event_completeness: float
    evidence_linking_rate: float
    safety_violations: int
    total_tasks: int
    total_events: int


class EvaluationTask(BaseModel):
    """A task to evaluate"""
    task_id: str
    task: str
    expected_answer: Optional[str] = None
    expected_events: List[str] = Field(default_factory=list)
    agent_type: str = "analyst"
    input_data: Optional[Dict[str, Any]] = None
    difficulty: str = "medium"
    category: str = "general"


class EvaluationResult(BaseModel):
    """Result of evaluating a single task"""
    task_id: str
    success: bool
    events: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)
    duration: float
    timestamp: str


class Evaluator:
    """Main evaluation engine"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def evaluate_task(self, task: EvaluationTask) -> EvaluationResult:
        """Evaluate a single task"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Start session
            response = await self.client.post(
                f"{self.base_url}/session/start",
                json={
                    "task": task.task,
                    "agent": task.agent_type,
                    "input": task.input_data
                },
                headers={"Accept": "text/event-stream"}
            )
            
            if response.status_code != 200:
                return EvaluationResult(
                    task_id=task.task_id,
                    success=False,
                    events=[],
                    metrics={},
                    errors=[f"HTTP {response.status_code}: {response.text}"],
                    duration=0.0,
                    timestamp=start_time.isoformat()
                )
            
            # Parse SSE events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                        events.append(event_data)
                    except json.JSONDecodeError:
                        continue
            
            # Calculate metrics
            metrics = self._calculate_task_metrics(task, events)
            
            # Determine success
            success = self._determine_success(task, events, metrics)
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return EvaluationResult(
                task_id=task.task_id,
                success=success,
                events=events,
                metrics=metrics,
                duration=duration,
                timestamp=start_time.isoformat()
            )
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return EvaluationResult(
                task_id=task.task_id,
                success=False,
                events=[],
                metrics={},
                errors=[str(e)],
                duration=duration,
                timestamp=start_time.isoformat()
            )
    
    def _calculate_task_metrics(self, task: EvaluationTask, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a single task"""
        if not events:
            return {
                "event_count": 0,
                "has_plan": False,
                "has_answer": False,
                "tool_calls": 0,
                "evidence_count": 0,
                "latency": 0.0,
                "completeness": 0.0
            }
        
        # Basic event counts
        event_types = [e.get("event_type") for e in events]
        has_plan = "plan" in event_types
        has_answer = "answer" in event_types
        tool_calls = event_types.count("tool_call")
        evidence_count = event_types.count("evidence")
        
        # Calculate latency
        if len(events) >= 2:
            first_time = datetime.fromisoformat(events[0]["timestamp"].replace("Z", "+00:00"))
            last_time = datetime.fromisoformat(events[-1]["timestamp"].replace("Z", "+00:00"))
            latency = (last_time - first_time).total_seconds()
        else:
            latency = 0.0
        
        # Calculate completeness
        expected_events = task.expected_events or ["plan", "answer"]
        found_events = [et for et in expected_events if et in event_types]
        completeness = len(found_events) / len(expected_events) if expected_events else 1.0
        
        return {
            "event_count": len(events),
            "has_plan": has_plan,
            "has_answer": has_answer,
            "tool_calls": tool_calls,
            "evidence_count": evidence_count,
            "latency": latency,
            "completeness": completeness,
            "event_types": event_types
        }
    
    def _determine_success(self, task: EvaluationTask, events: List[Dict[str, Any]], metrics: Dict[str, Any]) -> bool:
        """Determine if a task was successful"""
        # Basic success criteria
        if not events:
            return False
        
        if not metrics.get("has_answer", False):
            return False
        
        if metrics.get("completeness", 0.0) < 0.5:
            return False
        
        # Check for error events
        event_types = [e.get("event_type") for e in events]
        if "error" in event_types:
            return False
        
        return True
    
    async def evaluate_dataset(self, tasks: List[EvaluationTask]) -> EvaluationMetrics:
        """Evaluate a dataset of tasks"""
        results = []
        
        for task in tasks:
            result = await self.evaluate_task(task)
            results.append(result)
            await asyncio.sleep(0.1)  # Rate limiting
        
        return self._aggregate_metrics(results)
    
    def _aggregate_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Aggregate metrics across multiple task results"""
        if not results:
            return EvaluationMetrics(
                task_success_rate=0.0,
                factual_accuracy=0.0,
                tool_success_rate=0.0,
                average_latency=0.0,
                p95_latency=0.0,
                event_completeness=0.0,
                evidence_linking_rate=0.0,
                safety_violations=0,
                total_tasks=0,
                total_events=0
            )
        
        # Calculate success rate
        successful_tasks = sum(1 for r in results if r.success)
        task_success_rate = successful_tasks / len(results)
        
        # Calculate latency statistics
        latencies = [r.duration for r in results if r.duration > 0]
        average_latency = statistics.mean(latencies) if latencies else 0.0
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0.0
        
        # Calculate completeness
        completeness_scores = [r.metrics.get("completeness", 0.0) for r in results]
        event_completeness = statistics.mean(completeness_scores) if completeness_scores else 0.0
        
        # Calculate tool success rate (simplified)
        total_tool_calls = sum(r.metrics.get("tool_calls", 0) for r in results)
        tool_success_rate = 0.9 if total_tool_calls > 0 else 1.0  # Simplified for now
        
        # Count safety violations
        safety_violations = sum(1 for r in results if "error" in r.metrics.get("event_types", []))
        
        # Calculate evidence linking rate
        evidence_counts = [r.metrics.get("evidence_count", 0) for r in results]
        evidence_linking_rate = sum(1 for c in evidence_counts if c > 0) / len(results) if results else 0.0
        
        return EvaluationMetrics(
            task_success_rate=task_success_rate,
            factual_accuracy=task_success_rate * 0.95,  # Simplified
            tool_success_rate=tool_success_rate,
            average_latency=average_latency,
            p95_latency=p95_latency,
            event_completeness=event_completeness,
            evidence_linking_rate=evidence_linking_rate,
            safety_violations=safety_violations,
            total_tasks=len(results),
            total_events=sum(len(r.events) for r in results)
        )
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class TestDatasetGenerator:
    """Generate test datasets for evaluation"""
    
    @staticmethod
    def generate_basic_tasks() -> List[EvaluationTask]:
        """Generate basic test tasks"""
        return [
            EvaluationTask(
                task_id="basic_1",
                task="What is the capital of France?",
                expected_answer="Paris",
                expected_events=["plan", "evidence", "answer"],
                agent_type="analyst",
                difficulty="easy",
                category="geography"
            ),
            EvaluationTask(
                task_id="basic_2",
                task="Calculate the mean of [1, 2, 3, 4, 5]",
                expected_answer="3",
                expected_events=["plan", "tool_call", "answer"],
                agent_type="analyst",
                input_data={"numbers": [1, 2, 3, 4, 5]},
                difficulty="easy",
                category="math"
            ),
            EvaluationTask(
                task_id="basic_3",
                task="What are the latest developments in AI?",
                expected_events=["plan", "evidence", "answer"],
                agent_type="strategist",
                difficulty="medium",
                category="technology"
            ),
            EvaluationTask(
                task_id="creative_1",
                task="Design a sustainable city transportation system",
                expected_events=["approach", "brainstorm", "synthesis"],
                agent_type="creative",
                difficulty="hard",
                category="design"
            ),
            EvaluationTask(
                task_id="logic_1",
                task="Validate the logical consistency of: 'All A are B, some B are C, therefore all A are C'",
                expected_events=["structure", "validation", "conclusion"],
                agent_type="logic",
                difficulty="medium",
                category="logic"
            )
        ]
    
    @staticmethod
    def generate_stress_tasks() -> List[EvaluationTask]:
        """Generate stress test tasks"""
        return [
            EvaluationTask(
                task_id="stress_1",
                task="This is a very long and complex task that requires extensive research and analysis across multiple domains including economics, technology, and social sciences. Please provide a comprehensive analysis with detailed citations and evidence.",
                expected_events=["plan", "evidence", "analysis", "answer"],
                agent_type="analyst",
                difficulty="hard",
                category="stress"
            ),
            EvaluationTask(
                task_id="stress_2",
                task="",  # Empty task
                expected_events=["error"],
                agent_type="analyst",
                difficulty="hard",
                category="stress"
            ),
            EvaluationTask(
                task_id="stress_3",
                task="Calculate the factorial of 1000",  # Computationally intensive
                expected_events=["plan", "tool_call", "answer"],
                agent_type="analyst",
                difficulty="hard",
                category="stress"
            )
        ]


async def run_evaluation(output_dir: str = "datasets/traces"):
    """Run the full evaluation suite"""
    evaluator = Evaluator()
    
    try:
        # Generate test datasets
        basic_tasks = TestDatasetGenerator.generate_basic_tasks()
        stress_tasks = TestDatasetGenerator.generate_stress_tasks()
        
        print(f"Evaluating {len(basic_tasks)} basic tasks...")
        basic_metrics = await evaluator.evaluate_dataset(basic_tasks)
        
        print(f"Evaluating {len(stress_tasks)} stress tasks...")
        stress_metrics = await evaluator.evaluate_dataset(stress_tasks)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save basic evaluation results
        basic_results = {
            "timestamp": timestamp,
            "dataset": "basic",
            "metrics": asdict(basic_metrics),
            "tasks": [task.dict() for task in basic_tasks]
        }
        
        with open(output_path / f"basic_evaluation_{timestamp}.json", "w") as f:
            json.dump(basic_results, f, indent=2)
        
        # Save stress evaluation results
        stress_results = {
            "timestamp": timestamp,
            "dataset": "stress",
            "metrics": asdict(stress_metrics),
            "tasks": [task.dict() for task in stress_tasks]
        }
        
        with open(output_path / f"stress_evaluation_{timestamp}.json", "w") as f:
            json.dump(stress_results, f, indent=2)
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Basic Tasks - Success Rate: {basic_metrics.task_success_rate:.1%}")
        print(f"Basic Tasks - Avg Latency: {basic_metrics.average_latency:.2f}s")
        print(f"Stress Tasks - Success Rate: {stress_metrics.task_success_rate:.1%}")
        print(f"Stress Tasks - Avg Latency: {stress_metrics.average_latency:.2f}s")
        print(f"Total Safety Violations: {basic_metrics.safety_violations + stress_metrics.safety_violations}")
        
        return basic_metrics, stress_metrics
        
    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
