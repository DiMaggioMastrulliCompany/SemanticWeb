from typing import Any, Dict, TypedDict

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import re
import time
import time

load_dotenv()

# ==========================================
# 1. STATE DEFINITION (MEMORY)
# ==========================================

client = ChatOpenAI(model="mistralai/ministral-8b", base_url="https://openrouter.ai/api/v1")


class GraphState(TypedDict):
    query: str
    task_type: str  # hamilton, triangle, substructure...

    # Search state (stack & history)
    partial_solution: str  # The current stack (e.g. ['A', 'B', 'C'])
    forbidden_moves: str  # E.g. { 'A,B': ['C'] } -> if we're in [A,B], C is forbidden

    # Agent communication
    current_proposal: str  # The last proposed step
    verifier_feedback: str  # Feedback (VALID / INVALID)
    attempt_count: int  # Failed attempts at the current step

    # Final state
    final_output: str
    status: str  # 'SEARCHING', 'SOLVED', 'FAILED'


# ==========================================
# 2. PROMPT CONFIGURATION (DYNAMIC)
# ==========================================

TASK_SPECIFICS = {
    "hamilton": {
        "goal": "Find a Hamiltonian path (visiting every node exactly once).",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Partial Solution string (FULL path so far, e.g. '[0, 1, 2]');\n"
            "  2) 'BACKTRACK' if you need to back up;\n"
            "  3) The final answer: 'Yes, [path]' or 'No'."
        ),
        "solution_criteria": "The solution must be 'Yes, [path]' or 'No'."
    },
    "substructure": {
        "goal": "Determine if a subgraph is present in the graph.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Mapping string (FULL mapping so far, e.g. 'a->0, b->1');\n"
            "  2) 'BACKTRACK';\n"
            "  3) 'Yes' if found;\n"
            "  4) 'No' if not found."
        ),
        "solution_criteria": "The solution must be 'Yes' or 'No'."
    },
    "connectivity": {
        "goal": "Determine if two nodes are connected.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Path string (FULL path so far, e.g. '6-9-10');\n"
            "  2) 'Yes' if connected;\n"
            "  3) 'No' if not connected;\n"
            "  4) 'BACKTRACK'."
        ),
        "solution_criteria": "The final solution MUST be 'Yes' or 'No'. Paths are steps."
    },
    "cycle": {
        "goal": "Determine if there is a cycle in the graph.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Path string (FULL path explored so far);\n"
            "  2) 'Yes' if a cycle exists;\n"
            "  3) 'No' if no cycle exists;\n"
            "  4) 'BACKTRACK'."
        ),
        "solution_criteria": "The final solution MUST be 'Yes' or 'No'."
    },
    "flow": {
        "goal": "Find the maximum flow between two nodes.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Flow State (e.g. 'Current Flow: 2. Added path: 4->5->7');\n"
            "  2) The final maximum flow value (e.g. '5');\n"
            "  3) 'BACKTRACK'."
        ),
        "solution_criteria": "The final solution MUST be a single number representing the maximum flow. Intermediate flow updates are VALID STEPS."
    },
    "bipartite": {
        "goal": "Determine if the graph is bipartite.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Coloring (FULL coloring so far, e.g. '0:1, 1:2, 2:1');\n"
            "  2) 'Yes' if bipartite;\n"
            "  3) 'No' if not bipartite;\n"
            "  4) 'BACKTRACK'."
        ),
        "solution_criteria": "The final solution MUST be 'Yes' or 'No'. Coloring updates are VALID STEPS."
    },
    "triangle": {
        "goal": "Find the maximum sum of weights of three interconnected nodes (triangle).",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Best Triangle (e.g. 'Max so far: 17 (0-9-10)');\n"
            "  2) 'BACKTRACK';\n"
            "  3) The final maximum sum (e.g. '17')."
        ),
        "solution_criteria": "The solution must be the maximum sum value. Candidate triangles are VALID STEPS."
    },
    "shortest": {
        "goal": "Find the weight of the shortest path between two nodes.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Path/Distance (FULL path/dist so far);\n"
            "  2) 'BACKTRACK';\n"
            "  3) The final shortest path weight (e.g. '7')."
        ),
        "solution_criteria": "The solution must be a single number representing the total weight of the shortest path."
    },
    "topology": {
        "goal": "Find a topological sort of the graph.",
        "format": (
            "Output EXACTLY one of the following:\n"
            "  1) The UPDATED Partial Solution string (FULL sorted list so far);\n"
            "  2) 'BACKTRACK';\n"
            "  3) The final sorted list (e.g. '[0, 1, 2]')."
        ),
        "solution_criteria": "The solution must be a valid topological sort list (if u->v, u must appear BEFORE v)."
    }
}

UNIFIED_PROPOSER_TEMPLATE = (
    "You are a graph solver agent.\n"
    "Task Goal: {goal}\n"
    "Current Solution/State: {partial}\n"
    "Past Forbidden Moves: {forbidden}\n"
    "Instructions:\n"
    "{format}\n"
    "Do NOT include any extra commentary. The system will accept your returned string as the new partial solution when VALID."
)

UNIFIED_VERIFIER_TEMPLATE = (
    "You are a strict logic verifier.\n"
    "Task Goal: {goal}\n"
    "Input: Graph Description, Current State, and Proposal.\n"
    "Determine if the PROPOSAL is logically valid.\n"
    "Criteria for Final Solution: {solution_criteria}\n"
    "Instructions:\n"
    "- If the proposal matches the Final Solution Criteria EXACTLY, output 'VALID_SOLUTION'.\n"
    "- If the proposal is a valid partial step towards the solution (but NOT the final solution itself), output 'VALID_STEP'.\n"
    "- If it is a BACKTRACK request, output 'VALID_BACKTRACK'.\n"
    "- Otherwise, output 'INVALID: <reason>'.\n"
    "IMPORTANT: Do NOT mark a step as a solution. If the proposal is a path/mapping but the criteria requires 'Yes'/'No'/'FINISHED', output 'VALID_STEP'."
)

# ==========================================
# 3. NODE IMPLEMENTATION
# ==========================================


def call_model(messages, max_retries=5, long_output_retry=3, max_tokens=1000):
    """Calls the model with retry logic and guards against excessively long outputs.

    Behavior:
    - On exceptions: retry with exponential backoff up to `max_retries` attempts.
    - If the model returns content estimated to be longer than `max_tokens` (simple
      whitespace-token heuristic), retry the call up to `long_output_retry` attempts.
    - If after `long_output_retry` attempts the response is still too long, raise
      a `ValueError` to indicate the problem.
    """
    for attempt in range(max_retries):
        try:
            response = client.invoke(messages)
            content = response.content

            # Heuristic token estimate: split on whitespace
            token_est = len(str(content).split())
            if token_est > max_tokens:
                # Too long response; retry a limited number of times
                if attempt < long_output_retry:
                    time.sleep(1 + attempt)  # short backoff before retry
                    continue
                # Exceeded allowed long-output retries
                raise ValueError(f"Model response too long: ~{token_est} tokens (limit {max_tokens})")

            return content
        except Exception as e:
            # If this was caused by too-long response we raised ValueError above
            # For other exceptions, perform exponential backoff and retry
            if isinstance(e, ValueError):
                # propagate length errors immediately (already handled above by raising explicitly)
                raise

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff for transient errors
                continue

            # If it's the last attempt or a different error, re-raise
            if attempt == max_retries - 1:
                raise e
    return ""

def proposer_node(state: GraphState) -> Dict[str, Any]:
    task = state["task_type"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]
    manager_instruction = state.get("manager_instruction", "")

    specs = TASK_SPECIFICS.get(task, TASK_SPECIFICS["hamilton"])

    user_prompt = UNIFIED_PROPOSER_TEMPLATE.format(
        goal=specs["goal"],
        partial=str(partial),
        forbidden=str(forbidden),
        format=specs["format"]
    )

    # Include optional manager instruction so the model can react to backtrack or other control signals.
    if manager_instruction:
        user_prompt += f"\nManager Instruction: {manager_instruction}"

    msg = [
        ("system", f"You are a graph solver agent. Task: {task}. Graph Description:\n{state['query']}"),
        ("user", user_prompt),
    ]

    content = call_model(msg)

    proposal = str(content).strip().replace("'", "").replace('"', "")

    return {"current_proposal": proposal}


def verifier_node(state: GraphState) -> Dict[str, Any]:
    proposal = state["current_proposal"]

    # If the proposer requests a backtrack, the verifier implicitly approves the logical request
    if proposal == "BACKTRACK":
        return {"verifier_feedback": "VALID_BACKTRACK"}

    task = state["task_type"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]

    specs = TASK_SPECIFICS.get(task, TASK_SPECIFICS["hamilton"])

    user_prompt = UNIFIED_VERIFIER_TEMPLATE.format(
        goal=specs["goal"],
        solution_criteria=specs.get("solution_criteria", "Check if solution is complete.")
    )
    # Provide full context for the verifier to reason about the proposal
    user_prompt += f"\nCurrent Partial: {partial}\nProposal: {proposal}\nForbidden: {forbidden}"

    msg = [("system", f"You are a strict logic verifier. Graph Description:\n{state['query']}"), ("user", user_prompt)]

    content = call_model(msg)

    feedback = str(content).strip()

    # Output normalization
    if "VALID_SOLUTION" in feedback:
        return {"verifier_feedback": "VALID_SOLUTION"}
    elif "VALID_STEP" in feedback:
        return {"verifier_feedback": "VALID_STEP"}
    elif "VALID_BACKTRACK" in feedback:
        return {"verifier_feedback": "VALID_BACKTRACK"}
    elif feedback == "VALID": # Fallback for legacy or loose models
            return {"verifier_feedback": "VALID_STEP"}
    else:
        return {"verifier_feedback": feedback}


def manager_node(state: GraphState) -> Dict[str, Any]:
    """The backtracking manager: handles the search stack and control flow."""
    proposal = state["current_proposal"]
    feedback = state["verifier_feedback"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]

    # CASE 1: COMPLETE SUCCESS
    if feedback == "VALID_SOLUTION":
        return {"status": "SOLVED", "final_output": proposal}

    # CASE 2: VALID STEP (Progress)
    if feedback == "VALID_STEP":
        # Check for stagnation: if the proposal is identical to the current partial solution,
        # it means the agent is not advancing.
        if proposal == partial:
             # Treat as invalid to force a change
             log_line = f"STAGNATION: Proposal [{proposal}] is identical to Partial Solution. You must EXTEND or CHANGE it."
             new_forbidden = (forbidden + "\n" + log_line).strip()
             return {"forbidden_moves": new_forbidden, "attempt_count": state["attempt_count"] + 1, "status": "SEARCHING"}

        return {"partial_solution": proposal, "attempt_count": 0, "manager_instruction": "", "status": "SEARCHING"}

    # CASE 3: NEED TO BACKTRACK
    should_backtrack = (feedback == "VALID_BACKTRACK") or (state["attempt_count"] >= 3)

    if should_backtrack:
        # Check if we are stuck in a BACKTRACK loop (Proposer didn't update state)
        if state.get("manager_instruction") == "BACKTRACK" and proposal == "BACKTRACK":
             # Heuristic manual backtrack
             if "\n" in partial:
                 new_partial = partial.rsplit("\n", 1)[0]
             elif "," in partial:
                 new_partial = partial.rsplit(",", 1)[0]
             elif "->" in partial:
                 new_partial = partial.rsplit("->", 1)[0]
             elif " " in partial:
                 new_partial = partial.rsplit(" ", 1)[0]
             else:
                 new_partial = ""

             return {
                "partial_solution": new_partial,
                "attempt_count": 0,
                "manager_instruction": "",
                "status": "SEARCHING"
             }

        # If partial is empty, we cannot backtrack further
        if not partial or partial.strip() == "":
            return {"status": "FAILED", "final_output": "NO SOLUTION FOUND"}

        # Instead of manipulating structure, record a human-readable forbidden log line
        log_line = f"BACKTRACK_REQUEST from partial=[{partial}] forbids proposal=[{proposal}]"
        new_forbidden = (forbidden + "\n" + log_line).strip()

        # Ask the proposer to perform a backtrack by including a manager instruction
        return {
            "partial_solution": partial,
            "forbidden_moves": new_forbidden,
            "attempt_count": 0,
            "manager_instruction": "BACKTRACK",
            "status": "SEARCHING",
        }

    # CASE 4: SIMPLE ERROR (Retry the same step)
    # Append a human-readable forbidden note so the model can avoid repeating it
    log_line = f"INVALID_PROPOSAL at partial=[{partial}] -> proposal=[{proposal}] ; reason=[{feedback}]"
    new_forbidden = (forbidden + "\n" + log_line).strip()

    return {"forbidden_moves": new_forbidden, "attempt_count": state["attempt_count"] + 1, "status": "SEARCHING"}


def normalize_answer(raw_output: str, task_type: str) -> str:
    """Normalizes the model output for comparison."""
    raw = raw_output.strip()

    # Yes/No tasks
    if task_type in ["connectivity", "cycle", "bipartite", "substructure", "hamilton"]:
        # Look for explicit Yes/No at the start or end, or as a standalone word
        if re.search(r'\bYes\b', raw, re.IGNORECASE):
            return "Yes"
        if re.search(r'\bNo\b', raw, re.IGNORECASE):
            return "No"
        return raw # Return raw if ambiguous

    # Numeric tasks
    if task_type in ["flow", "shortest", "triangle"]:
        # Extract the last number found in the string
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
        if numbers:
            return numbers[-1]
        return raw

    # List/Path tasks (Topology)
    if task_type == "topology":
        # Try to find a list pattern
        match = re.search(r'\[(.*?)\]', raw)
        if match:
            return f"[{match.group(1)}]"
        return raw

    return raw

def verify_answer(prediction: str, ground_truth: str, task_type: str) -> bool:
    """Verifies if the prediction matches the ground truth."""
    pred_norm = normalize_answer(prediction, task_type)
    gt_norm = normalize_answer(ground_truth, task_type)

    # Direct comparison for normalized values
    if pred_norm.lower() == gt_norm.lower():
        return True

    # Special case for Hamilton: "Yes, [path]" vs "Yes"
    if task_type == "hamilton":
        if pred_norm == "Yes" and "Yes" in gt_norm:
            return True

    # Special case for Substructure: "No solution" often means "No"
    if task_type == "substructure":
        if prediction == "No solution" and gt_norm == "No":
            return True

    return False

def parser_node(state: GraphState) -> Dict[str, Any]:
    # Format the final output according to the dataset requirements
    if state["status"] == "FAILED":
        return {"final_output": "No solution"}

    raw = state["final_output"]
    if raw == "NO SOLUTION FOUND":
        return {"final_output": "No solution"}

    # Normalize the output
    normalized = normalize_answer(raw, state["task_type"])

    # If we had access to ground truth in the state, we could verify here.
    # Since we don't always have it in the state, we just return the normalized output
    # or the raw output + normalized version.

    return {"final_output": normalized}


# ==========================================
# 4. GRAPH AND ROUTING
# ==========================================


def router(state: GraphState) -> str:
    if state["status"] == "SOLVED":
        return "parser"
    if state["status"] == "FAILED":
        return "parser"
    return "proposer"  # Continue the loop (forward/backward is handled by the state)


def build_backtracking_solver(recursion_limit: int = 50):
    workflow = StateGraph(GraphState)

    workflow.add_node("proposer", proposer_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("manager", manager_node)
    workflow.add_node("parser", parser_node)

    workflow.set_entry_point("proposer")

    workflow.add_edge("proposer", "verifier")
    workflow.add_edge("verifier", "manager")

    workflow.add_conditional_edges("manager", router, {"proposer": "proposer", "parser": "parser"})

    workflow.add_edge("parser", END)
    return workflow.compile()


# ==========================================
# 5. DEMO RUN
# ==========================================

if __name__ == "__main__":
    # Hamiltonian example
    # A-B, B-C, A-C. Path: A->B->C
    graphwiz_dataset = load_dataset("GraphWiz/GraphInstruct")

    example = graphwiz_dataset["train"][0]

    print("EXAMPLE QUERY:")
    print(example["query"])
    print("EXAMPLE TASK:")
    print(example["task"])
    print("EXAMPLE SOLUTION:")
    print(example["answer"])

    initial_state: GraphState = {
        "query": str(example["query"]),
        "task_type": str(example["task"]),
        "partial_solution": "",  # Starts empty; the proposer will suggest the first node
        "forbidden_moves": "",
        "current_proposal": "",
        "verifier_feedback": "",
        "attempt_count": 0,
        "final_output": "",
        "status": "SEARCHING",
    }

    app = build_backtracking_solver()

    # Run with streaming to observe intermediate steps
    for step in app.stream(initial_state, config={"recursion_limit": 100}):
        print(step)
