import copy
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from trace_logger import TraceLogger, Step, StepType
from causal_attribution import CausalAttribution
from llm_client import LLMClient
from text_processor import convert_text_to_jsonl
from semantic_minimality import Embedder
from utils import (
    extract_step_text,
    calculate_minimality_lex,
    calculate_minimality_edit,
    calculate_minimality_sem,
)


class Repair:
    def __init__(
        self,
        step_id: int,
        original_step: Step,
        repaired_step: Step,
        minimality_lex: float,
        success_predicted: bool,
        repaired_trace: Optional["TraceLogger"] = None,
        minimality_edit: Optional[float] = None,
        minimality_sem: Optional[float] = None,
        original_text: Optional[str] = None,
        repaired_text: Optional[str] = None,
        proposal_idx: Optional[int] = None,
    ):

        self.step_id = step_id #The step being repaired
        self.original_step = original_step #The original (faulty) step
        self.repaired_step = repaired_step #The repaired step
        self.minimality_lex = minimality_lex #Lexical minimality
        self.minimality_edit = minimality_edit #Normalized char-level Levenshtein minimality
        self.minimality_sem = minimality_sem #Embedding-cosine minimality, in [0, 1]
        self.success_predicted = success_predicted #Whether this repair is predicted to succeed
        self.repaired_trace = repaired_trace #Full trace after applying repair (only for successful repairs with reexecutor)
        self.original_text = original_text
        self.repaired_text = repaired_text
        self.proposal_idx = proposal_idx

    @property
    def minimality_score(self) -> float:
        """Backward-compat alias for the paper-default lexical score."""
        return self.minimality_lex

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "step_id": self.step_id,
            "original_step": self.original_step.to_dict(),
            "repaired_step": self.repaired_step.to_dict(),
            "minimality_score": self.minimality_lex,
            "minimality_lex": self.minimality_lex,
            "minimality_edit": self.minimality_edit,
            "minimality_sem": self.minimality_sem,
            "original_text": self.original_text,
            "repaired_text": self.repaired_text,
            "proposal_idx": self.proposal_idx,
            "success_predicted": self.success_predicted
        }
        # Include full repaired trace for successful repairs
        if self.success_predicted and self.repaired_trace is not None:
            result["repaired_trace"] = self.repaired_trace.to_dict()
        return result

class CounterfactualRepair:
    def __init__(
        self,
        trace: TraceLogger,
        causal_attribution: CausalAttribution,
        llm_client: LLMClient,
        reexecutor: Optional[Any] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        num_proposals: int = 3,
        compute_semantic_minimality: bool = False,
    ):

        self.trace = trace #The failed execution trace
        self.causal_attribution = causal_attribution #Causal attribution analysis results
        self.llm_client = llm_client
        self.reexecutor = reexecutor #Optional reexecutor for deterministic repair evaluation
        self.execution_context = execution_context or {} #Context needed for execution (prompt, tests, entry_point, etc.)
        self.num_proposals = num_proposals #Default K for proposal generation (paper default 3; ablation uses 5)
        self.compute_semantic_minimality = compute_semantic_minimality #Whether to call the embedder per proposal
        self._embedder: Optional[Embedder] = (
            Embedder() if compute_semantic_minimality else None
        )

        self.repairs: Dict[int, List[Repair]] = {}

    def generate_repairs(
        self,
        step_ids: List[int],
        num_proposals: Optional[int] = None,
    ) -> Dict[int, List[Repair]]:

        k = num_proposals if num_proposals is not None else self.num_proposals
        for step_id in step_ids:
            self.repairs[step_id] = self._generate_repairs_for_step(
                step_id,
                k,
                self.trace.steps
            )

        return self.repairs

    def _generate_repairs_for_step(
        self,
        step_id: int,
        num_proposals: int,
        all_steps: List[Step]
    ) -> List[Repair]:
        original_step = self.trace.get_step(step_id)
        if not original_step:
            return []

        proposals = []
        previous_steps = all_steps[:step_id]
        for i in range(num_proposals):
            prompt = self._create_repair_prompt(original_step, previous_steps)

            try:
                result = self.llm_client.generate_structured(
                    prompt,
                    schema_name="repair",
                    system_message="""You are an expert at debugging and fixing agent failures. 
                                    Your goal is to fix the specific LOGICAL ERROR in THIS step only, not to achieve the final answer.
                                    
                                    CRITICAL RULES:
                                    1. Fix what went wrong in THIS step's logic - focus on the error in THIS step, not the entire solution
                                    2. DO NOT directly use, search for, or reference the gold answer in your repair
                                    3. Generate MINIMAL, TARGETED edits that fix the logical error in THIS step
                                    4. For tool calls: fix the reasoning/query for THIS sub-goal, don't jump to searching for the final answer
                                    5. Your repair should make sense even if the gold answer were different
                                    
                                    Make the smallest possible change that fixes the logical error in THIS step.
                                    Always respond using the provided schema in JSON format.
                                    """,
                    temperature=0.7,
                    model_name=self.llm_client.model
                )

                repaired_step = self._apply_repair(original_step, result)

                original_text = extract_step_text(original_step)
                repaired_text = extract_step_text(repaired_step)

                minimality_lex = calculate_minimality_lex(original_text, repaired_text)
                minimality_edit = calculate_minimality_edit(original_text, repaired_text)
                minimality_sem: Optional[float] = None
                if self.compute_semantic_minimality and self._embedder is not None:
                    try:
                        minimality_sem = calculate_minimality_sem(
                            original_text, repaired_text, self._embedder
                        )
                    except Exception as e:
                        print(f"Semantic minimality failed for step {step_id} proposal {i}: {e}")
                        minimality_sem = None

                # Evaluate repair success deterministically if reexecutor is available, otherwise predict
                success_predicted, repaired_trace = self._evaluate_repair_success(
                    step_id, repaired_step, execution_context=self.execution_context
                )

                repair = Repair(
                    step_id=step_id,
                    original_step=original_step,
                    repaired_step=repaired_step,
                    minimality_lex=minimality_lex,
                    minimality_edit=minimality_edit,
                    minimality_sem=minimality_sem,
                    success_predicted=success_predicted,
                    repaired_trace=repaired_trace,
                    original_text=original_text,
                    repaired_text=repaired_text,
                    proposal_idx=i,
                )

                proposals.append(repair)

            except Exception as e:
                print(f"Error generating repair proposal {i} for step {step_id}: {e}")
                continue

        proposals.sort(key=lambda r: r.minimality_lex, reverse=True)

        return proposals

    def _create_repair_prompt(self, step: Step, previous_steps: List[Step]) -> str:
        execution_logs = self._extract_execution_logs()
        
        prompt = f"""You are a "Causal Debugger" for an AI Agent. 
The agent failed a task. Your goal is to fix the specific *logic error* in the current step.

Problem Statement: {self.trace.problem_statement}
Correct Answer (FOR REFERENCE ONLY): {self.trace.gold_answer}
Agent's Incorrect Answer: {self.trace.final_answer}

IMPORTANT: The correct answer is provided for reference to understand what the agent should have eventually arrived at.
DO NOT directly incorporate the gold answer into your repair. Your repair should fix the logical error in THIS step only.
"""

        if execution_logs:
            prompt += f"""
EXECUTION ERROR LOGS (what went wrong):
{execution_logs}

Use these error logs to understand the specific failure and create a targeted fix.
"""

        prompt += f"""
CRITICAL CONSTRAINTS:
1. The gold answer is provided for reference ONLY - DO NOT directly use it in your repair
2. Fix the logical error in THIS step only - not the entire solution path
3. For tool calls: fix the reasoning/query for THIS sub-goal, don't jump to searching for the final answer
4. Your repair should make sense even if the gold answer were different
5. Focus on what went wrong in THIS step's logic, not on achieving the correct final answer

BAD EXAMPLE: If this step searches for "Person B who won Rollo Davidson Prize", DO NOT change it to search for "Russell Lyons" (the gold answer)
GOOD EXAMPLE: Fix the search query logic to correctly find Person B based on the constraints given in THIS step

Recent previous steps:
{json.dumps([step.to_dict() for step in previous_steps], indent=2)}

TARGET STEP TO FIX: Below is the step that has been identified as causally responsible for the failure.

Step {step.step_id} ({step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Original Reasoning: {step.text}\n\n"
            prompt += "Based on the execution error logs above, provide a MINIMAL, TARGETED correction to fix the logical error in THIS reasoning step.\n"
            prompt += "The correction should directly address what went wrong in THIS step's logic, not jump to the final solution.\n"
            prompt += "Focus on fixing the reasoning flaw in THIS step only.\n"
            prompt += "Fill in 'repaired_text' with the corrected reasoning.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal and targeted in 'minimality_justification'."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Original Arguments: {json.dumps(step.tool_args)}\n\n"
            prompt += "Based on the execution error logs, identify and fix the logical error in THIS tool call.\n"
            prompt += "Focus on fixing what went wrong with THIS specific tool usage - the reasoning or query for THIS sub-goal.\n"
            prompt += "DO NOT change the tool arguments to directly search for or reference the gold answer.\n"
            prompt += "Example: If searching for 'Person B', fix the search logic for Person B - don't change it to search for the final answer.\n"
            prompt += "Fill in 'repaired_tool_name' and 'repaired_tool_args_json' (as a valid JSON string, e.g. '{\"key\": \"value\"}').\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal and targeted in 'minimality_justification'."

        elif step.step_type == StepType.LLM_RESPONSE:
            prompt += f"Original LLM Response (code): {step.text}\n\n"
            prompt += "Based on the execution error logs, provide a MINIMAL, TARGETED code fix for the logical error in THIS step.\n"
            prompt += "The fix should directly address the specific error shown in the logs (e.g., fix the exact line causing the error).\n"
            prompt += "Focus on fixing what went wrong in THIS step's code logic, not on producing the final answer.\n"
            prompt += "Fill in 'repaired_text' with the corrected code.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal and targeted in 'minimality_justification'."

        elif step.step_type == StepType.MEMORY_ACCESS:
            prompt += f"Memory Key: {step.memory_key}\n"
            prompt += f"Original Value: {step.memory_value}\n\n"
            prompt += "Based on the execution error logs, fix the logical error in this memory access step.\n"
            prompt += "Focus on what went wrong with THIS memory operation, not on achieving the final answer.\n"
            prompt += "Fill in 'repaired_text' with the corrected value.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        elif step.step_type == StepType.TOOL_RESPONSE:
            prompt += f"Tool Output: {step.tool_output}\n"
            prompt += f"Tool Call Result: {step.tool_call_result}\n\n"
            prompt += "Based on the execution error logs, fix the logical error in this tool response.\n"
            prompt += "Focus on what went wrong with THIS tool's output, not on producing the final answer.\n"
            prompt += "Fill in 'repaired_text' with the corrected output.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        else:
            content = step.text or step.action or step.observation
            prompt += f"Original Content: {content}\n\n"
            prompt += "Based on the execution error logs, provide a minimal, targeted correction to fix the logical error in THIS step.\n"
            prompt += "Focus on what went wrong in THIS step, not on achieving the final answer.\n"
            prompt += "Fill in 'repaired_text'.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        prompt += """

CRITICAL REMINDERS:
1. You are patching the track, not driving the train - fix THIS step's logic, not the entire solution
2. DO NOT search for, reference, or incorporate the gold answer in your repair
3. If you simply state the final answer, the repair will be rejected for low minimality
4. Fix only the logical error in THIS step - your repair should work regardless of what the final answer is

CONCRETE EXAMPLES OF BAD vs GOOD REPAIRS:
BAD: Changing search query from "Rollo Davidson Prize winner" to "Russell Lyons PhD 1983" (jumps to final answer)
GOOD: Changing search query from "Rollo Davidson Prize winner" to "Rollo Davidson Prize 1990-2005" (fixes the logic for THIS step)

BAD: Updating reasoning to "The answer is Russell Lyons" (states final answer)
GOOD: Updating reasoning to "Need to check publication dates between 1990-2005, not all publications" (fixes logic flaw)
"""

        return prompt

    def _extract_execution_logs(self) -> str:
        """Extract execution error logs from the trace."""
        logs_parts: List[str] = []
        
        # First, try to get logs from execution_context
        if self.execution_context and "logs" in self.execution_context:
            logs = self.execution_context.get("logs")
            if logs and isinstance(logs, str) and logs.strip():
                logs_parts.append(logs.strip())
        
        # Also extract from TOOL_RESPONSE steps in the trace
        for step in self.trace.steps:
            if step.step_type == StepType.TOOL_RESPONSE:
                if step.tool_output and isinstance(step.tool_output, str):
                    output = step.tool_output.strip()
                    if output and output not in logs_parts:
                        logs_parts.append(output)
        
        if not logs_parts:
            return "No execution logs available."
        
        return "\n".join(logs_parts)

    def _apply_repair(self, original_step: Step, repair_result: BaseModel) -> Step:
        repaired_step = copy.deepcopy(original_step)

        if original_step.step_type == StepType.REASONING:
            repaired_step.text = repair_result.repaired_text or original_step.text

        elif original_step.step_type == StepType.TOOL_CALL:
            # Parse tool args from JSON string
            if repair_result.repaired_tool_args_json is not None:
                parsed = convert_text_to_jsonl(repair_result.repaired_tool_args_json)
                if not parsed:
                    raise ValueError(f"Failed to parse repaired_tool_args_json: {repair_result.repaired_tool_args_json}")
                repaired_step.tool_args = parsed[0]
            if repair_result.repaired_tool_name is not None:
                repaired_step.tool_name = repair_result.repaired_tool_name

        elif original_step.step_type == StepType.LLM_RESPONSE:
            if repair_result.repaired_text is not None:
                repaired_step.text = repair_result.repaired_text

        elif original_step.step_type == StepType.MEMORY_ACCESS:
            repaired_step.memory_value = repair_result.repaired_text or original_step.memory_value

        elif original_step.step_type == StepType.ENVIRONMENT_ACTION:
            repaired_step.action = repair_result.repaired_text or original_step.action

        elif original_step.step_type == StepType.TOOL_RESPONSE:
            # For tool responses, repair goes into tool_output
            if repair_result.repaired_text is not None:
                repaired_step.tool_output = repair_result.repaired_text
            # Keep original tool_output if no repair provided

        else:
            repaired_step.text = repair_result.repaired_text or original_step.text

        return repaired_step

    def _evaluate_repair_success(
        self, step_id: int, repaired_step: Step, execution_context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[TraceLogger]]:
        """
        Evaluate repair success by re-executing with the repaired step.

        If an agent is provided, build a history up to and including the repaired
        step, then call agent.run_remaining_steps() to continue execution and
        observe the actual outcome.

        Returns:
            A tuple of (success_predicted, repaired_trace).
            repaired_trace is only populated for successful repairs when a reexecutor is available.
        """
        if self.reexecutor is None:
            success = self.causal_attribution._llm_predict_outcome(
                step_id, repaired_step, self.trace.problem_statement
            )
            return (success, None)

        # Build history: all steps before step_id + the repaired step
        history = [copy.deepcopy(step) for step in self.trace.steps if step.step_id < step_id]
        history.append(repaired_step)

        new_trace = self.reexecutor.run_remaining_steps(history)
        # Return trace only for successful repairs to save storage
        if new_trace.success:
            return (True, new_trace)
        return (False, None)

    def get_successful_repairs(self, step_id: int) -> List[Repair]:
        if step_id not in self.repairs:
            return []

        successful = [r for r in self.repairs[step_id] if r.success_predicted]
        return successful

    def get_all_successful_repairs(self) -> Dict[int, List[Repair]]:

        successful_repairs = {}

        for step_id in self.repairs.keys():
            repairs = self.get_successful_repairs(step_id)
            if repairs is not None:
                successful_repairs[step_id] = repairs
        
        return successful_repairs