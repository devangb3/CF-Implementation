from typing import Optional

from trace_logger import Step, StepType
from semantic_minimality import cosine_similarity, Embedder



def summarize_step(step: Step) -> str:

    if step.step_type == StepType.REASONING:
        return f"[Reasoning] {step.text}"

    elif step.step_type == StepType.TOOL_CALL:
        args_str = str(step.tool_args) if step.tool_args else ""
        return f"[Tool Call] {step.tool_name}({args_str})"

    elif step.step_type == StepType.LLM_RESPONSE:
        return f"[LLM Response] {step.text}"

    elif step.step_type == StepType.TOOL_RESPONSE:
        output_str = str(step.tool_output)
        return f"[Tool Response] {output_str}"

    elif step.step_type == StepType.MEMORY_ACCESS:
        return f"[Memory Access] Key: {step.memory_key}, Value: {step.memory_value}"

    elif step.step_type == StepType.ENVIRONMENT_ACTION:
        return f"[Environment Action] {step.action}"

    elif step.step_type == StepType.ENVIRONMENT_OBSERVATION:
        return f"[Environment Observation] {step.observation}"

    elif step.step_type == StepType.FINAL_ANSWER:
        return f"[Final Answer] {step.text}"

    else:
        return f"[Unknown Step Type] {step.step_type}"


def extract_step_text(step: Step) -> str:

    if step.step_type == StepType.REASONING:
        return step.text or ""

    elif step.step_type == StepType.TOOL_CALL:
        # For tool calls, format as: tool_name(args)
        args_str = str(step.tool_args) if step.tool_args else ""
        return f"{step.tool_name}({args_str})"

    elif step.step_type == StepType.TOOL_RESPONSE:
        return str(step.tool_output) if step.tool_output else ""

    elif step.step_type == StepType.MEMORY_ACCESS:
        return f"Memory[{step.memory_key}] = {step.memory_value}"

    elif step.step_type == StepType.ENVIRONMENT_ACTION:
        return step.action or ""

    elif step.step_type == StepType.ENVIRONMENT_OBSERVATION:
        return step.observation or ""

    elif step.step_type == StepType.FINAL_ANSWER:
        return step.text or ""

    return ""


def format_step_context(step: Step, context: str) -> str:
    step_summary = summarize_step(step)

    if context:
        return f"""Step ID: {step.step_id}
Type: {step.step_type.value}
Content: {step_summary}

Context from dependencies:
{context}"""
    else:
        return f"""Step ID: {step.step_id}
Type: {step.step_type.value}
Content: {step_summary}"""


def calculate_text_similarity(text1: str, text2: str) -> float:

    # Simple word-based tokenization
    tokens1 = set[str](text1.lower().split())
    tokens2 = set[str](text2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    # Jaccard similarity
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def calculate_minimality_lex(original: str, modified: str) -> float:

    if original == modified:
        return 1.0

    original_tokens = original.lower().split()
    modified_tokens = modified.lower().split()

    if not original_tokens:
        return 0.0 if modified_tokens else 1.0

    # Calculate the proportion of unchanged tokens
    max_len = max(len(original_tokens), len(modified_tokens))

    # Count matching tokens (simple approach)
    matches = sum(1 for i in range(min(len(original_tokens), len(modified_tokens)))
                  if original_tokens[i] == modified_tokens[i])

    # Additional penalty for length difference
    len_diff_penalty = abs(len(original_tokens) - len(modified_tokens)) / max_len

    minimality = (matches / max_len) * (1 - len_diff_penalty * 0.5)

    return max(0.0, min(1.0, minimality))


calculate_minimality_score = calculate_minimality_lex


def calculate_minimality_edit(original: str, modified: str) -> float:
    if original == modified:
        return 1.0
    if not original and not modified:
        return 1.0

    a, b = original, modified
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    distance = prev[lb]
    return max(0.0, 1.0 - distance / max(la, lb))


def calculate_minimality_sem(
    original: str,
    modified: str,
    embedder: Embedder,
) -> float:
    if original == modified:
        return 1.0
    if not original and not modified:
        return 1.0
    if not original or not modified:
        return 0.0
   
    vecs = embedder.embed([original, modified])
    cos = cosine_similarity(vecs[0], vecs[1])
    # Map [-1, 1] -> [0, 1]
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))
