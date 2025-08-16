# JuniorGPT - Visible Thinking AI Platform

A production-ready, multi-agent AI system that performs structured reasoning with visible thinking, evidence gathering, and tool integration. Built for transparency, auditability, and enterprise compliance.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Architecture**: 4 specialized agents (Analyst, Strategist, Creative, Logic)
- **Visible Thinking**: Real-time streaming of reasoning steps via Server-Sent Events
- **Tool Integration**: Search, mathematical analysis, data validation
- **Schema Validation**: All events validated against JSON schemas
- **Evaluation Framework**: Automated testing and metrics collection
- **Modern Web UI**: Beautiful, responsive interface for interaction

### Agents
- **Analyst**: Data analysis, evidence gathering, fact-checking
- **Strategist**: Strategic planning, high-level reasoning, research
- **Creative**: Creative problem-solving, ideation, innovation
- **Logic**: Logical reasoning, argument validation, contradiction detection

### Tools
- **Search**: Web search using DuckDuckGo API
- **Math**: Mathematical calculations and statistical analysis
- **Validate**: Data validation and quality assessment

## 🏗️ Architecture

```
JuniorGPT/
├── app/
│   ├── main.py          # FastAPI orchestrator
│   ├── agents.py        # Agent framework
│   ├── tools.py         # Tool framework
│   └── static/          # Web UI
├── contracts/
│   └── events.schema.json
├── models/
│   └── policy.yaml
├── scripts/
│   ├── eval/           # Evaluation framework
│   └── validate_schemas.py
├── datasets/
│   └── traces/         # Evaluation results
└── safety/
    └── redact_check.py
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd junior-gpt-crisp-dm
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key if using API mode
   ```

4. **Run the system**
   ```bash
   make run
   ```

## 🎯 Quick Start

### Web Interface
1. Start the server: `make run`
2. Open http://localhost:8001 in your browser
3. Select an agent, enter a task, and watch the visible thinking unfold!

### API Usage
```bash
# Start a session
curl -X POST http://localhost:8001/session/start \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"task": "What is the capital of France?", "agent": "analyst"}'

# List available agents
curl http://localhost:8001/agents

# List available tools
curl http://localhost:8001/tools
```

### Demo Session
```bash
make demo
```

## 📊 Evaluation

Run the comprehensive evaluation suite:

```bash
make eval
```

This will:
- Test all agents with basic and stress scenarios
- Generate performance metrics
- Save results to `datasets/traces/`

## 🔧 Configuration

### Agent Policy (`models/policy.yaml`)
```yaml
backends:
  api:
    provider: openai
    model: gpt-4o-mini

routing:
  default_agent: analyst
  analyst: api
  strategist: api
  creative: api
  logic: api
```

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_MODEL`: Model to use (default: gpt-4o-mini)
- `MODEL_BACKEND`: Backend type (api/local)

## 📈 Performance Metrics

The system tracks:
- **Task Success Rate**: Percentage of tasks completed successfully
- **Factual Accuracy**: Accuracy of claims and citations
- **Tool Success Rate**: Success rate of tool invocations
- **Latency**: Response times (average and P95)
- **Event Completeness**: Completeness of reasoning traces
- **Safety Violations**: Number of safety incidents

## 🧪 Testing

```bash
# Run tests
make test

# Validate schemas
make validate

# Check system health
make health
```

## 🚀 Deployment

### Development
```bash
make dev  # Debug mode with hot reload
```

### Production
```bash
make prod  # Production server with multiple workers
```

### Docker (coming soon)
```bash
docker build -t juniorgpt .
docker run -p 8001:8001 juniorgpt
```

## 📚 API Reference

### Endpoints

#### `POST /session/start`
Start a new reasoning session.

**Request:**
```json
{
  "task": "Your task or question",
  "agent": "analyst|strategist|creative|logic",
  "input": {"optional": "structured data"}
}
```

**Response:** Server-Sent Events stream with reasoning steps

#### `GET /agents`
List all available agents and their capabilities.

#### `GET /tools`
List all available tools and their schemas.

#### `POST /tools/{tool_name}/execute`
Execute a specific tool directly.

### Event Types

- `plan`: Strategic planning steps
- `evidence`: Evidence gathering results
- `analysis`: Data analysis results
- `answer`: Final answer
- `error`: Error events
- `approach`: Creative approach definition
- `brainstorm`: Brainstorming results
- `synthesis`: Creative synthesis
- `structure`: Logical structure analysis
- `validation`: Validation results
- `conclusion`: Logical conclusions

## 🔒 Safety & Compliance

- **Schema Validation**: All events validated against JSON schemas
- **Input Sanitization**: Tool inputs validated and sanitized
- **Error Handling**: Comprehensive error handling and logging
- **Audit Trail**: Complete trace of all reasoning steps
- **Privacy**: No sensitive data stored by default

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: See `JuniorGPT-CRISP-DM.md` for detailed CRISP-DM methodology
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussions**: Use GitHub discussions for questions and ideas

## 🎯 Roadmap

- [ ] Local model support (LLaMA, Mistral)
- [ ] Additional agents (Researcher, Validator, Executor, etc.)
- [ ] Advanced tool integrations
- [ ] Training pipeline for fine-tuning
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Advanced evaluation metrics
- [ ] Multi-language support

---

**Built with ❤️ for transparent, auditable AI reasoning**
