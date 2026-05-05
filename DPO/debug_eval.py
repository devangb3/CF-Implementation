import re

def strip_fences(text):
    return re.sub(r'```\w*\n?', '', text).strip()

# Pick first code pair from test set
code_pair = next(p for p in test_pairs if p['step_type'] == 'llm_response')

# Build prompt exactly as generate_repair does
prompt_text = f"Problem: {code_pair['problem_statement']}\n"
if code_pair['prior_context'].strip():
    prompt_text += f"Context: {code_pair['prior_context'].strip()}\n"
prompt_text += f"Failing step:\n{strip_fences(code_pair['rejected'])}\nOutput only the corrected code, no explanation:"

messages = [{'role': 'user', 'content': prompt_text}]

print("=== PROMPT (as seen by model) ===")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# Generate
generated = generate_repair(eval_model, tokenizer, code_pair)

print("\n=== GENERATED (raw) ===")
print(repr(generated[:800]))

print("\n=== CHOSEN (target) ===")
print(repr(code_pair['chosen'][:800]))

print("\n=== EXEC TEST ===")
task_id = int(code_pair['problem_id'].replace('mbpp-', ''))
tests = mbpp_tests.get(task_id, [])
extracted = extract_code(generated)

print("Extracted code:")
print(extracted)
print("\nTests:")
for t in tests:
    print(f"  {t}")

# Run each test assertion individually to pinpoint failures
print("\nPer-test results:")
import traceback
namespace = {}
try:
    exec(compile(extracted, '<string>', 'exec'), namespace)
    for t in tests:
        try:
            exec(t, namespace.copy())
            print(f"  PASS  {t}")
        except Exception as e:
            print(f"  FAIL  {t}")
            print(f"        {e}")
except Exception as e:
    print(f"  Syntax/compile error in extracted code:")
    traceback.print_exc()

print("\nOverall:", exec_test(extracted, tests))

# ── Reasoning pair debug ───────────────────────────────────────────────────
print("\n\n" + "="*60)
print("REASONING PAIR DEBUG")
print("="*60)

reasoning_pair = next(p for p in test_pairs if p['step_type'] == 'reasoning')

prompt_text_r = f"Problem: {reasoning_pair['problem_statement']}\n"
if reasoning_pair['prior_context'].strip():
    prompt_text_r += f"Context: {reasoning_pair['prior_context'].strip()}\n"
prompt_text_r += f"Failing step:\n{reasoning_pair['rejected']}\nOutput only the corrected reasoning, no explanation:"

messages_r = [{'role': 'user', 'content': prompt_text_r}]

print("\n=== PROMPT (as seen by model) ===")
print(tokenizer.apply_chat_template(messages_r, tokenize=False, add_generation_prompt=True))

generated_r = generate_repair(eval_model, tokenizer, reasoning_pair)

print("\n=== GENERATED (raw) ===")
print(repr(generated_r[:800]))

print("\n=== CHOSEN (target) ===")
print(repr(reasoning_pair['chosen'][:800]))

print("\n=== REJECTED (original) ===")
print(repr(reasoning_pair['rejected'][:800]))

from difflib import SequenceMatcher
sim_chosen   = SequenceMatcher(None, generated_r, reasoning_pair['chosen']).ratio()
sim_rejected = SequenceMatcher(None, generated_r, reasoning_pair['rejected']).ratio()
print(f"\nSimilarity to chosen  : {sim_chosen:.3f}")
print(f"Similarity to rejected: {sim_rejected:.3f}  (should be lower than chosen)")
