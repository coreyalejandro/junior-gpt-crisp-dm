import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, Field

# Import our agent and tool frameworks
from .agents import AgentContext, agent_registry
from .tools import tool_registry

# Optional: load environment from .env if present
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

# Load policy configuration
POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "policy.yaml")
DEFAULT_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "api")

policy: Dict[str, Any] = {
	"backends": {
		"api": {"provider": "openai", "model": DEFAULT_MODEL}
	},
	"routing": {
		"default_agent": "analyst",
		"analyst": "api",
		"strategist": "api",
		"creative": "api",
		"logic": "api",
	},
}
if os.path.exists(POLICY_PATH):
	with open(POLICY_PATH, "r", encoding="utf-8") as f:
		loaded = yaml.safe_load(f) or {}
		# Shallow merge with env override precedence
		policy.update({k: {**policy.get(k, {}), **v} for k, v in loaded.items()})
		# Ensure model respects env override
		if "backends" in policy and "api" in policy["backends"]:
			policy["backends"]["api"]["model"] = os.getenv("OPENAI_API_MODEL", policy["backends"]["api"].get("model", DEFAULT_MODEL))

# Load event schema
SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "contracts")
EVENT_SCHEMA_PATH = os.path.join(SCHEMA_DIR, "events.schema.json")
with open(EVENT_SCHEMA_PATH, "r", encoding="utf-8") as f:
	event_schema = json.load(f)
_event_validator = Draft202012Validator(event_schema)


def validate_event(event: Dict[str, Any]) -> None:
	"""Validate an event against the JSON schema, raising HTTP 500 on error."""
	try:
		_event_validator.validate(event)
	except ValidationError as e:
		raise HTTPException(status_code=500, detail=f"Event schema validation failed: {e.message}")


class TaskInput(BaseModel):
	task: str = Field(..., description="User task or question")
	input: Optional[Dict[str, Any]] = Field(default=None, description="Structured input payload for tools")
	agent: str = Field(default=policy.get("routing", {}).get("default_agent", "analyst"), description="Agent type")


app = FastAPI(title="JuniorGPT Orchestrator", version="0.1.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
	app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/health")
async def health() -> Dict[str, str]:
	return {"status": "ok", "backend": MODEL_BACKEND}


@app.get("/tools")
async def list_tools() -> Dict[str, Any]:
	"""List all available tools"""
	return {
		"tools": tool_registry.list_tools(),
		"total": len(tool_registry.tools)
	}


@app.get("/agents")
async def list_agents() -> Dict[str, Any]:
	"""List all available agents"""
	return {
		"agents": agent_registry.list_agents(),
		"total": len(agent_registry.agents)
	}


@app.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
	"""Execute a specific tool"""
	from .tools import ToolCall
	
	tool_call = ToolCall(
		tool_name=tool_name,
		parameters=parameters,
		trace_id=str(uuid.uuid4())
	)
	
	result = await tool_registry.execute_tool(tool_call)
	return result.dict()


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _sse_format(event_type: str, data_obj: Dict[str, Any]) -> str:
	return f"event: {event_type}\n" f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"


@app.post("/session/start")
async def start_session(body: TaskInput) -> StreamingResponse:
	"""Start a session and stream visible-thinking events via SSE."""
	session_id = str(uuid.uuid4())
	
	# Validate agent type
	agent_type = body.agent
	if agent_type not in agent_registry.agents:
		agent_type = policy.get("routing", {}).get("default_agent", "analyst")
	
	# Create agent context
	context = AgentContext(
		session_id=session_id,
		task=body.task,
		input_data=body.input
	)
	
	async def event_stream() -> AsyncGenerator[bytes, None]:
		try:
			# Execute the selected agent
			async for agent_event in agent_registry.execute_agent(agent_type, context):
				# Convert agent event to our standard format
				event = {
					"session_id": agent_event.session_id,
					"agent_type": agent_event.agent_type,
					"event_type": agent_event.event_type,
					"payload": agent_event.payload,
					"timestamp": agent_event.timestamp,
				}
				
				# Validate the event
				validate_event(event)
				
				# Stream the event
				yield _sse_format(agent_event.event_type, event).encode("utf-8")
				await asyncio.sleep(0)  # Allow flush
				
		except Exception as e:
			# Send error event
			error_event = {
				"session_id": session_id,
				"agent_type": agent_type,
				"event_type": "error",
				"payload": {"error": str(e), "type": "execution_error"},
				"timestamp": _now_iso(),
			}
			validate_event(error_event)
			yield _sse_format("error", error_event).encode("utf-8")
	
	return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/session/{session_id}/continue")
async def continue_session(session_id: str, body: Dict[str, Any]) -> StreamingResponse:
	"""Continue an existing session with additional input or clarification."""
	
	# For now, we'll start a new session with the continuation
	# In a full implementation, this would load session history
	task = body.get("task", "Continue previous session")
	agent_type = body.get("agent", policy.get("routing", {}).get("default_agent", "analyst"))
	
	# Create new context (in full implementation, would include history)
	context = AgentContext(
		session_id=session_id,
		task=task,
		input_data=body.get("input"),
		metadata={"continuation": True}
	)
	
	async def event_stream() -> AsyncGenerator[bytes, None]:
		try:
			async for agent_event in agent_registry.execute_agent(agent_type, context):
				event = {
					"session_id": agent_event.session_id,
					"agent_type": agent_event.agent_type,
					"event_type": agent_event.event_type,
					"payload": agent_event.payload,
					"timestamp": agent_event.timestamp,
				}
				
				validate_event(event)
				yield _sse_format(agent_event.event_type, event).encode("utf-8")
				await asyncio.sleep(0)
				
		except Exception as e:
			error_event = {
				"session_id": session_id,
				"agent_type": agent_type,
				"event_type": "error",
				"payload": {"error": str(e), "type": "continuation_error"},
				"timestamp": _now_iso(),
			}
			validate_event(error_event)
			yield _sse_format("error", error_event).encode("utf-8")
	
	return StreamingResponse(event_stream(), media_type="text/event-stream")


# Fallback for non-existing routes
@app.exception_handler(404)
async def not_found_handler(_, __):
	return JSONResponse(status_code=404, content={"error": "Not Found"})
