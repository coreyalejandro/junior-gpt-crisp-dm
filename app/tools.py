import asyncio
import json
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import httpx
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from a tool execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool call request"""
    tool_name: str
    parameters: Dict[str, Any]
    trace_id: str


class BaseTool:
    """Base class for all tools"""
    
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.description = "Base tool description"
        self.parameters_schema = {}
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters"""
        raise NotImplementedError


class SearchTool(BaseTool):
    """Web search tool using DuckDuckGo API"""
    
    def __init__(self):
        super().__init__()
        self.name = "search"
        self.description = "Search the web for current information"
        self.parameters_schema = {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5, "description": "Maximum number of results"}
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 5)
            
            if not query:
                return ToolResult(success=False, error="Query parameter is required")
            
            # Use DuckDuckGo Instant Answer API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant information
                    results = []
                    
                    # Add instant answer if available
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", "Instant Answer"),
                            "snippet": data.get("Abstract"),
                            "url": data.get("AbstractURL", ""),
                            "source": "DuckDuckGo Instant Answer"
                        })
                    
                    # Add related topics
                    for topic in data.get("RelatedTopics", [])[:max_results-1]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                "snippet": topic.get("Text"),
                                "url": topic.get("FirstURL", ""),
                                "source": "DuckDuckGo Related Topics"
                            })
                    
                    return ToolResult(
                        success=True,
                        data={
                            "query": query,
                            "results": results[:max_results],
                            "total_results": len(results)
                        },
                        metadata={"source": "DuckDuckGo API"}
                    )
                else:
                    return ToolResult(success=False, error=f"Search API returned status {response.status_code}")
                    
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {str(e)}")


class MathTool(BaseTool):
    """Mathematical analysis tool"""
    
    def __init__(self):
        super().__init__()
        self.name = "math"
        self.description = "Perform mathematical calculations and statistical analysis"
        self.parameters_schema = {
            "operation": {"type": "string", "enum": ["calculate", "statistics", "correlation"], "description": "Type of operation"},
            "expression": {"type": "string", "description": "Mathematical expression or data"},
            "data": {"type": "array", "items": {"type": "number"}, "description": "Array of numbers for statistical analysis"}
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            operation = parameters.get("operation", "calculate")
            
            if operation == "calculate":
                expression = parameters.get("expression", "")
                if not expression:
                    return ToolResult(success=False, error="Expression parameter required for calculate operation")
                
                # Safe evaluation of mathematical expressions
                allowed_names = {
                    'abs': abs, 'round': round, 'min': min, 'max': max,
                    'sum': sum, 'len': len, 'pow': pow, 'sqrt': math.sqrt,
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'log': math.log, 'log10': math.log10, 'exp': math.exp,
                    'pi': math.pi, 'e': math.e
                }
                
                try:
                    result = eval(expression, {"__builtins__": {}}, allowed_names)
                    return ToolResult(
                        success=True,
                        data={"result": result, "expression": expression}
                    )
                except Exception as e:
                    return ToolResult(success=False, error=f"Invalid expression: {str(e)}")
            
            elif operation == "statistics":
                data = parameters.get("data", [])
                if not data:
                    return ToolResult(success=False, error="Data array required for statistics operation")
                
                try:
                    stats = {
                        "count": len(data),
                        "sum": sum(data),
                        "mean": statistics.mean(data),
                        "median": statistics.median(data),
                        "std_dev": statistics.stdev(data) if len(data) > 1 else 0,
                        "min": min(data),
                        "max": max(data),
                        "range": max(data) - min(data)
                    }
                    
                    return ToolResult(
                        success=True,
                        data={"statistics": stats, "data": data}
                    )
                except Exception as e:
                    return ToolResult(success=False, error=f"Statistical analysis failed: {str(e)}")
            
            elif operation == "correlation":
                data1 = parameters.get("data1", [])
                data2 = parameters.get("data2", [])
                
                if len(data1) != len(data2) or len(data1) < 2:
                    return ToolResult(success=False, error="Two arrays of equal length (>=2) required for correlation")
                
                try:
                    correlation = statistics.correlation(data1, data2)
                    return ToolResult(
                        success=True,
                        data={
                            "correlation": correlation,
                            "data1": data1,
                            "data2": data2,
                            "interpretation": self._interpret_correlation(correlation)
                        }
                    )
                except Exception as e:
                    return ToolResult(success=False, error=f"Correlation calculation failed: {str(e)}")
            
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(success=False, error=f"Math tool failed: {str(e)}")
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r >= 0.9:
            strength = "very strong"
        elif abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.5:
            strength = "moderate"
        elif abs_r >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} correlation"


class DataValidationTool(BaseTool):
    """Data validation and quality assessment tool"""
    
    def __init__(self):
        super().__init__()
        self.name = "validate"
        self.description = "Validate data quality and structure"
        self.parameters_schema = {
            "data": {"type": "object", "description": "Data to validate"},
            "schema": {"type": "object", "description": "Expected schema"},
            "checks": {"type": "array", "items": {"type": "string"}, "description": "Types of checks to perform"}
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            data = parameters.get("data", {})
            schema = parameters.get("schema", {})
            checks = parameters.get("checks", ["completeness", "types", "ranges"])
            
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "summary": {}
            }
            
            # Completeness check
            if "completeness" in checks:
                missing_fields = []
                for field in schema.get("required", []):
                    if field not in data:
                        missing_fields.append(field)
                        validation_results["valid"] = False
                
                if missing_fields:
                    validation_results["errors"].append(f"Missing required fields: {missing_fields}")
                
                validation_results["summary"]["completeness"] = {
                    "total_fields": len(schema.get("required", [])),
                    "missing_fields": len(missing_fields),
                    "completeness_rate": (len(schema.get("required", [])) - len(missing_fields)) / len(schema.get("required", [])) if schema.get("required") else 1.0
                }
            
            # Type checking
            if "types" in checks:
                type_errors = []
                for field, expected_type in schema.get("properties", {}).items():
                    if field in data:
                        actual_type = type(data[field]).__name__
                        if expected_type.get("type") and actual_type != expected_type["type"]:
                            type_errors.append(f"{field}: expected {expected_type['type']}, got {actual_type}")
                            validation_results["valid"] = False
                
                if type_errors:
                    validation_results["errors"].append(f"Type mismatches: {type_errors}")
            
            # Range/constraint checking
            if "ranges" in checks:
                range_errors = []
                for field, constraints in schema.get("properties", {}).items():
                    if field in data:
                        value = data[field]
                        
                        if "minimum" in constraints and value < constraints["minimum"]:
                            range_errors.append(f"{field}: value {value} below minimum {constraints['minimum']}")
                            validation_results["valid"] = False
                        
                        if "maximum" in constraints and value > constraints["maximum"]:
                            range_errors.append(f"{field}: value {value} above maximum {constraints['maximum']}")
                            validation_results["valid"] = False
                
                if range_errors:
                    validation_results["errors"].append(f"Range violations: {range_errors}")
            
            return ToolResult(
                success=True,
                data=validation_results
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Validation failed: {str(e)}")


class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of tools"""
        default_tools = [
            SearchTool(),
            MathTool(),
            DataValidationTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their schemas"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_schema": tool.parameters_schema
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call"""
        tool = self.get_tool(tool_call.tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_call.tool_name}' not found")
        
        return await tool.execute(tool_call.parameters)


# Global tool registry instance
tool_registry = ToolRegistry()
