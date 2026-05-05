# CausalFlow Examples

This directory contains example scripts demonstrating how to use the CausalFlow framework for analyzing agent failures.

## Overview

CausalFlow is a framework for:
- **Causal Attribution**: Identifying which steps in an agent's execution chain caused failures
- **Counterfactual Repair**: Generating repairs to fix failed reasoning chains
- **Multi-Agent Critique**: Using multiple critic agents to evaluate and improve agent performance

## Examples

### 1. Simple Example: `demo.py`

**Difficulty**: Beginner

A basic demonstration showing how to create a failed trace and analyze it with CausalFlow.

**Problem**: John has 5 apples and buys 3 more. How many apples does he have?
- **Expected Answer**: 8
- **Agent's Answer**: 6 (incorrect)
- **Error Type**: Agent fabricates information not present in the problem

**Key Features**:
- 5 execution steps
- Simple arithmetic failure
- Easy-to-understand error pattern
- Good for understanding CausalFlow basics

**Usage**:
```bash
cd /path/to/CausalFlow
python examples/demo.py
```

**Functions**:
- `create_sample_failed_trace()`: Creates a trace with a simple reasoning error
- `demo_basic_trace()`: Demonstrates trace logging without CausalFlow analysis
- `demo_causal_flow_analysis()`: Runs full CausalFlow analysis
- `demo_with_real_agent()`: Shows integration with a real LLM-powered agent

---

### 2. Complex Example: `complex_example.py`

**Difficulty**: Advanced

A more realistic failure scenario demonstrating CausalFlow's ability to trace errors through longer reasoning chains.

**Problem**: A store has 120 items in stock. They sell 35% in the morning, receive a shipment of 40 items, then sell 25 more items. How many items remain?
- **Expected Answer**: 93
- **Agent's Answer**: 95 (incorrect)
- **Error Type**: Incorrect rounding during intermediate calculation

**Key Features**:
- 14 execution steps (much longer chain)
- Multi-step problem with percentages
- Error propagation through correct operations
- Realistic agent failure pattern

**Problem Breakdown**:
```
Morning sales: 35% of 120 = 42 items
After morning: 120 - 42 = 78
After shipment: 78 + 40 = 118
After afternoon sales: 118 - 25 = 93 ✓

Agent's Error (Step 4):
- Correctly calculates 42
- Incorrectly rounds to 40 "for simplicity"
- All subsequent calculations are technically correct
- But final answer is wrong due to the early error
```

**Why This Example is Better**:
1. **Longer reasoning chains**: Demonstrates CausalFlow's ability to track dependencies through many steps
2. **Error propagation**: Shows how a single mistake can cascade through otherwise correct logic
3. **Realistic failures**: Represents actual failure modes seen in LLM agents
4. **Multiple operations**: Includes multiplication, subtraction, and addition

**Usage**:
```bash
cd /path/to/CausalFlow
python examples/complex_example.py
```

**Output Files**:
- `examples/complex_trace.json`: The execution trace
- `examples/complex_analysis_report.txt`: Human-readable analysis report
- `examples/complex_analysis_results.json`: Structured analysis results

---

### 3. Example Agent: `example_agent.py`

A reusable `MathReasoningAgent` class that demonstrates how to integrate TraceLogger into your own agents.

**Features**:
- Chain-of-thought reasoning
- Tool calls (calculator)
- Automatic trace logging
- Integration with LLM client

**Usage**:
```python
from llm_client import LLMClient
from example_agent import MathReasoningAgent

# Create agent
llm_client = LLMClient(api_key="your-api-key")
agent = MathReasoningAgent(llm_client)

# Solve a problem
answer = agent.solve(
    problem="Sarah has 12 cookies and eats 4. How many are left?",
    gold_answer="8"
)

# Get the trace
trace = agent.get_trace()

# Analyze if it failed
if not trace.success:
    flow = CausalFlow(api_key="your-api-key")
    results = flow.analyze_trace(trace)
```

## Prerequisites

Before running these examples, ensure you have:

1. **API Key**: Set up your OpenRouter key in a `.env` file in the project root. The codebase reads **`OPENROUTER_SECRET_KEY`** (see root `.env.example`).
   ```bash
   OPENROUTER_SECRET_KEY=your-api-key-here
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running Examples

### Run Simple Example
```bash
python examples/demo.py
```

### Run Complex Example
```bash
python examples/complex_example.py
```

## What You'll Learn

### From Simple Example (`demo.py`)
- Basic trace logging with `TraceLogger`
- Creating step dependencies
- Recording outcomes (success/failure)
- Running CausalFlow analysis
- Generating reports

### From Complex Example (`complex_example.py`)
- Handling multi-step reasoning chains
- Tracking error propagation
- Analyzing complex failure patterns
- Understanding causal attribution scores
- Interpreting counterfactual repairs

### From Example Agent (`example_agent.py`)
- Integrating TraceLogger into your agents
- Logging different step types (reasoning, tool calls, tool responses)
- Managing step dependencies
- Saving and loading traces

## Output Files

After running the examples, you'll find generated files in the `examples/` directory:

- **Traces**: `*.json` files containing execution traces
- **Reports**: `*_report.txt` files with human-readable analysis
- **Results**: `*_results.json` files with structured analysis data

## Next Steps

1. **Modify the examples**: Try changing the problems or error patterns
2. **Create your own agent**: Use `example_agent.py` as a template
3. **Analyze real failures**: Apply CausalFlow to your own agent failures
4. **Experiment with parameters**: Adjust CausalFlow settings for different analyses

## Comparison: Simple vs Complex Example

| Feature | Simple Example | Complex Example |
|---------|---------------|-----------------|
| **Steps** | 5 | 14 |
| **Operations** | 1 (addition) | 3 (multiplication, addition, subtraction) |
| **Error Type** | Fabrication | Incorrect rounding |
| **Difficulty** | Beginner | Advanced |
| **Use Case** | Learning basics | Realistic failures |
| **Runtime** | ~30 seconds | ~60 seconds |

## Troubleshooting

**Problem**: `OPENROUTER_SECRET_KEY` missing or “not found”
- **Solution**: Create a `.env` file in the project root with your API key

**Problem**: `ModuleNotFoundError`
- **Solution**: Run from the project root directory, not from within `examples/`

**Problem**: API errors
- **Solution**: Check your API key and internet connection

## Additional Resources

- [CausalFlow paper PDF](../research/CausalFlow.pdf): manuscript describing the method (§4–§5 match the implemented pipeline)
- [Main README](../README.md): Project overview and setup
- [OpenRouter Documentation](https://openrouter.ai/docs): LLM API details
