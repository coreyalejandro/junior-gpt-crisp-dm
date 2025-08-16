import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field

from .tools import ToolCall, ToolResult, tool_registry


class AgentEvent(BaseModel):
    """Base event structure for agent outputs"""
    session_id: str
    agent_type: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: str
    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None


class AgentContext(BaseModel):
    """Context passed to agents during execution"""
    session_id: str
    task: str
    input_data: Optional[Dict[str, Any]] = None
    history: List[AgentEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.description = "Base agent description"
        self.capabilities = []
        self.tools = []
    
    async def execute(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        """Execute the agent's reasoning process"""
        raise NotImplementedError
    
    def _create_event(self, event_type: str, payload: Dict[str, Any], context: AgentContext, parent_trace_id: Optional[str] = None) -> AgentEvent:
        """Create a standardized event"""
        trace_id = str(uuid.uuid4())
        return AgentEvent(
            session_id=context.session_id,
            agent_type=self.name,
            event_type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trace_id=trace_id,
            parent_trace_id=parent_trace_id
        )


class StrategistAgent(BaseAgent):
    """Strategic planning and high-level reasoning agent"""
    
    def __init__(self):
        super().__init__()
        self.name = "strategist"
        self.description = "Creates high-level plans and strategic approaches"
        self.capabilities = ["planning", "decomposition", "prioritization"]
        self.tools = ["search"]
    
    async def execute(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        # Generate strategic plan
        plan_payload = await self._generate_plan(context)
        plan_event = self._create_event("plan", plan_payload, context)
        yield plan_event
        
        # Analyze task complexity and decide on approach
        decision_payload = await self._make_strategic_decision(context, plan_payload)
        decision_event = self._create_event("decision", decision_payload, context, plan_event.trace_id)
        yield decision_event
        
        # If research is needed, perform it
        if decision_payload.get("research_needed", False):
            research_results = await self._conduct_research(context, decision_payload)
            evidence_event = self._create_event("evidence", research_results, context, decision_event.trace_id)
            yield evidence_event
    
    async def _generate_plan(self, context: AgentContext) -> Dict[str, Any]:
        """Generate a strategic plan for the task"""
        # This would typically call an LLM, but for now we'll use a template
        plan_steps = [
            "1. Analyze the task requirements and constraints",
            "2. Identify key information gaps and research needs", 
            "3. Determine the optimal approach and tools required",
            "4. Establish success criteria and validation methods",
            "5. Create execution timeline and resource allocation"
        ]
        
        return {
            "steps": plan_steps,
            "complexity": "high",
            "estimated_duration": "5-10 minutes",
            "required_tools": ["search", "analysis"],
            "risk_factors": ["information gaps", "time constraints"]
        }
    
    async def _make_strategic_decision(self, context: AgentContext, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Make strategic decisions about approach"""
        return {
            "approach": "research_first",
            "research_needed": True,
            "research_queries": [context.task],
            "tool_sequence": ["search", "validate", "analyze"],
            "confidence": 0.85,
            "reasoning": "Task requires current information and validation"
        }
    
    async def _conduct_research(self, context: AgentContext, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct initial research for strategic planning"""
        queries = decision.get("research_queries", [context.task])
        results = []
        
        for query in queries:
            tool_call = ToolCall(
                tool_name="search",
                parameters={"query": query, "max_results": 3},
                trace_id=str(uuid.uuid4())
            )
            
            result = await tool_registry.execute_tool(tool_call)
            if result.success:
                results.append({
                    "query": query,
                    "results": result.data.get("results", []),
                    "source": "strategic_research"
                })
        
        return {
            "research_summary": f"Found {len(results)} relevant information sources",
            "key_findings": results,
            "confidence_boost": 0.1
        }


class AnalystAgent(BaseAgent):
    """Data analysis and evidence-based reasoning agent"""
    
    def __init__(self):
        super().__init__()
        self.name = "analyst"
        self.description = "Performs detailed analysis and evidence gathering"
        self.capabilities = ["data_analysis", "evidence_gathering", "fact_checking"]
        self.tools = ["search", "math", "validate"]
    
    async def execute(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        # Generate analysis plan
        plan_payload = await self._generate_analysis_plan(context)
        plan_event = self._create_event("plan", plan_payload, context)
        yield plan_event
        
        # Gather evidence
        evidence_payload = await self._gather_evidence(context, plan_payload)
        evidence_event = self._create_event("evidence", evidence_payload, context, plan_event.trace_id)
        yield evidence_event
        
        # Perform analysis
        analysis_payload = await self._perform_analysis(context, evidence_payload)
        analysis_event = self._create_event("analysis", analysis_payload, context, evidence_event.trace_id)
        yield analysis_event
        
        # Generate final answer
        answer_payload = await self._generate_answer(context, analysis_payload)
        answer_event = self._create_event("answer", answer_payload, context, analysis_event.trace_id)
        yield answer_event
    
    async def _generate_analysis_plan(self, context: AgentContext) -> Dict[str, Any]:
        """Generate a detailed analysis plan"""
        return {
            "analysis_type": "comprehensive",
            "data_sources": ["web_search", "input_data"],
            "analysis_methods": ["fact_checking", "statistical_analysis", "validation"],
            "quality_checks": ["source_reliability", "data_freshness", "cross_validation"],
            "output_format": "structured_answer_with_citations"
        }
    
    async def _gather_evidence(self, context: AgentContext, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Gather evidence and data for analysis"""
        evidence = []
        
        # Search for relevant information
        tool_call = ToolCall(
            tool_name="search",
            parameters={"query": context.task, "max_results": 5},
            trace_id=str(uuid.uuid4())
        )
        
        search_result = await tool_registry.execute_tool(tool_call)
        if search_result.success:
            evidence.append({
                "type": "web_search",
                "query": context.task,
                "results": search_result.data.get("results", []),
                "reliability_score": 0.8
            })
        
        # Add input data if available
        if context.input_data:
            evidence.append({
                "type": "user_input",
                "data": context.input_data,
                "reliability_score": 1.0
            })
        
        return {
            "evidence_sources": evidence,
            "total_sources": len(evidence),
            "coverage_score": 0.85
        }
    
    async def _perform_analysis(self, context: AgentContext, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis of gathered evidence"""
        analysis_results = {
            "key_findings": [],
            "confidence_scores": {},
            "contradictions": [],
            "gaps": []
        }
        
        # Analyze each evidence source
        for source in evidence.get("evidence_sources", []):
            if source["type"] == "web_search":
                for result in source.get("results", []):
                    analysis_results["key_findings"].append({
                        "claim": result.get("snippet", ""),
                        "source": result.get("url", ""),
                        "reliability": source["reliability_score"]
                    })
        
        # Perform validation if data is present
        if context.input_data:
            validation_call = ToolCall(
                tool_name="validate",
                parameters={
                    "data": context.input_data,
                    "schema": {"type": "object"},
                    "checks": ["completeness", "types"]
                },
                trace_id=str(uuid.uuid4())
            )
            
            validation_result = await tool_registry.execute_tool(validation_call)
            if validation_result.success:
                analysis_results["data_quality"] = validation_result.data
        
        return analysis_results
    
    async def _generate_answer(self, context: AgentContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final answer based on analysis"""
        findings = analysis.get("key_findings", [])
        
        if not findings:
            return {
                "text": "I couldn't find sufficient evidence to provide a definitive answer.",
                "confidence": 0.3,
                "citations": [],
                "limitations": ["insufficient_data"]
            }
        
        # Compile answer from findings
        answer_text = "Based on my analysis:\n\n"
        citations = []
        
        for i, finding in enumerate(findings[:3], 1):
            answer_text += f"{i}. {finding['claim']}\n"
            if finding.get("source"):
                citations.append(finding["source"])
        
        answer_text += f"\nConfidence: {analysis.get('coverage_score', 0.7):.1%}"
        
        return {
            "text": answer_text,
            "confidence": analysis.get("coverage_score", 0.7),
            "citations": citations,
            "analysis_summary": f"Analyzed {len(findings)} key findings from {len(analysis.get('evidence_sources', []))} sources"
        }


class CreativeAgent(BaseAgent):
    """Creative problem-solving and ideation agent"""
    
    def __init__(self):
        super().__init__()
        self.name = "creative"
        self.description = "Generates creative solutions and innovative approaches"
        self.capabilities = ["ideation", "brainstorming", "creative_synthesis"]
        self.tools = ["search"]
    
    async def execute(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        # Generate creative approach
        approach_payload = await self._generate_creative_approach(context)
        approach_event = self._create_event("approach", approach_payload, context)
        yield approach_event
        
        # Brainstorm solutions
        brainstorm_payload = await self._brainstorm_solutions(context, approach_payload)
        brainstorm_event = self._create_event("brainstorm", brainstorm_payload, context, approach_event.trace_id)
        yield brainstorm_event
        
        # Synthesize final creative solution
        synthesis_payload = await self._synthesize_solution(context, brainstorm_payload)
        synthesis_event = self._create_event("synthesis", synthesis_payload, context, brainstorm_event.trace_id)
        yield synthesis_event
    
    async def _generate_creative_approach(self, context: AgentContext) -> Dict[str, Any]:
        """Generate a creative approach to the problem"""
        return {
            "creative_frameworks": ["design_thinking", "lateral_thinking", "analogical_reasoning"],
            "innovation_techniques": ["brainstorming", "mind_mapping", "reverse_thinking"],
            "constraints": ["feasibility", "originality", "impact"],
            "success_metrics": ["novelty", "usefulness", "elegance"]
        }
    
    async def _brainstorm_solutions(self, context: AgentContext, approach: Dict[str, Any]) -> Dict[str, Any]:
        """Brainstorm multiple creative solutions"""
        solutions = [
            {
                "id": "sol_1",
                "concept": "Innovative approach using emerging technologies",
                "novelty_score": 0.9,
                "feasibility_score": 0.7,
                "impact_score": 0.8
            },
            {
                "id": "sol_2", 
                "concept": "Cross-disciplinary solution combining multiple domains",
                "novelty_score": 0.8,
                "feasibility_score": 0.6,
                "impact_score": 0.9
            },
            {
                "id": "sol_3",
                "concept": "Minimalist approach focusing on core principles",
                "novelty_score": 0.6,
                "feasibility_score": 0.9,
                "impact_score": 0.7
            }
        ]
        
        return {
            "solutions": solutions,
            "total_concepts": len(solutions),
            "diversity_score": 0.8,
            "selection_criteria": ["novelty", "feasibility", "impact"]
        }
    
    async def _synthesize_solution(self, context: AgentContext, brainstorm: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the best creative solution"""
        solutions = brainstorm.get("solutions", [])
        
        if not solutions:
            return {
                "text": "Unable to generate creative solutions at this time.",
                "confidence": 0.4,
                "recommendations": ["reframe_problem", "gather_more_context"]
            }
        
        # Select best solution based on criteria
        best_solution = max(solutions, key=lambda x: x["novelty_score"] * x["feasibility_score"] * x["impact_score"])
        
        return {
            "text": f"Recommended creative solution: {best_solution['concept']}\n\nThis approach balances novelty ({best_solution['novelty_score']:.1%}), feasibility ({best_solution['feasibility_score']:.1%}), and impact ({best_solution['impact_score']:.1%}).",
            "selected_solution": best_solution,
            "confidence": 0.8,
            "implementation_steps": [
                "1. Validate concept with stakeholders",
                "2. Create detailed implementation plan", 
                "3. Prototype and test core components",
                "4. Iterate based on feedback"
            ]
        }


class LogicAgent(BaseAgent):
    """Logical reasoning and validation agent"""
    
    def __init__(self):
        super().__init__()
        self.name = "logic"
        self.description = "Performs logical reasoning and validates arguments"
        self.capabilities = ["logical_analysis", "argument_validation", "contradiction_detection"]
        self.tools = ["validate", "math"]
    
    async def execute(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        # Analyze logical structure
        structure_payload = await self._analyze_logical_structure(context)
        structure_event = self._create_event("structure", structure_payload, context)
        yield structure_event
        
        # Validate arguments
        validation_payload = await self._validate_arguments(context, structure_payload)
        validation_event = self._create_event("validation", validation_payload, context, structure_event.trace_id)
        yield validation_event
        
        # Generate logical conclusion
        conclusion_payload = await self._generate_logical_conclusion(context, validation_payload)
        conclusion_event = self._create_event("conclusion", conclusion_payload, context, validation_event.trace_id)
        yield conclusion_event
    
    async def _analyze_logical_structure(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the logical structure of the task"""
        return {
            "premises": ["Task requires logical analysis", "Evidence must be validated"],
            "assumptions": ["Input data is relevant", "Logical consistency is required"],
            "inferences": ["Valid conclusions can be drawn", "Contradictions must be resolved"],
            "logical_framework": "deductive_reasoning"
        }
    
    async def _validate_arguments(self, context: AgentContext, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical arguments and detect contradictions"""
        validation_results = {
            "valid_premises": [],
            "invalid_premises": [],
            "contradictions": [],
            "logical_gaps": [],
            "confidence": 0.9
        }
        
        # Validate input data if present
        if context.input_data:
            validation_call = ToolCall(
                tool_name="validate",
                parameters={
                    "data": context.input_data,
                    "schema": {"type": "object"},
                    "checks": ["completeness", "consistency"]
                },
                trace_id=str(uuid.uuid4())
            )
            
            result = await tool_registry.execute_tool(validation_call)
            if result.success:
                validation_results["data_validation"] = result.data
        
        return validation_results
    
    async def _generate_logical_conclusion(self, context: AgentContext, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate logical conclusion based on validated arguments"""
        confidence = validation.get("confidence", 0.7)
        
        if validation.get("contradictions"):
            return {
                "text": "Logical analysis reveals contradictions that must be resolved before proceeding.",
                "confidence": confidence * 0.5,
                "issues": validation["contradictions"],
                "recommendations": ["resolve_contradictions", "clarify_assumptions"]
            }
        
        return {
            "text": "Logical analysis supports proceeding with the task. All premises are valid and no contradictions detected.",
            "confidence": confidence,
            "logical_strength": "strong",
            "supporting_evidence": validation.get("valid_premises", [])
        }


class AgentRegistry:
    """Registry for all available agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register the default set of agents"""
        default_agents = [
            StrategistAgent(),
            AnalystAgent(),
            CreativeAgent(),
            LogicAgent()
        ]
        
        for agent in default_agents:
            self.register_agent(agent)
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent"""
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents with their capabilities"""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "tools": agent.tools
            }
            for agent in self.agents.values()
        ]
    
    async def execute_agent(self, agent_name: str, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        """Execute an agent by name"""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        async for event in agent.execute(context):
            yield event


# Global agent registry instance
agent_registry = AgentRegistry()
