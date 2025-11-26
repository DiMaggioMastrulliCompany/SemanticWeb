from typing import Any, Dict, TypedDict

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. DEFINIZIONE DELLO STATO (MEMORY)
# ==========================================

client = ChatOpenAI(model="google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1")


class GraphState(TypedDict):
    query: str
    task_type: str  # hamilton, triangle, substructure...

    # Stato della ricerca (Stack & History)
    partial_solution: str  # Lo stack corrente (es. ['A', 'B', 'C'])
    forbidden_moves: str  # Es. { "A,B": ["C"] } -> Se siamo in [A,B], C è bannato

    # Comunicazione Agenti
    current_proposal: str  # L'ultimo step proposto
    verifier_feedback: str  # Feedback (VALID / INVALID)
    attempt_count: int  # Tentativi falliti allo step corrente

    # Stato Finale
    final_output: str
    status: str  # 'SEARCHING', 'SOLVED', 'FAILED'


# ==========================================
# 2. CONFIGURAZIONE PROMPT (DINAMICI)
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
# 3. IMPLEMENTAZIONE NODI
# ==========================================


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

    response = client.invoke(msg)
    content = response.content

    proposal = str(content).strip().replace("'", "").replace('"', "")

    return {"current_proposal": proposal}


def verifier_node(state: GraphState) -> Dict[str, Any]:
    proposal = state["current_proposal"]

    # Se il proposer vuole fare backtrack, il verifier approva implicitamente la richiesta logica
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

    response = client.invoke(msg)
    content = response.content

    feedback = str(content).strip()

    # Normalizzazione output
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
    """Il cervello del Backtracking: gestisce lo Stack."""
    proposal = state["current_proposal"]
    feedback = state["verifier_feedback"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]

    # CASO 1: SUCCESSO TOTALE
    if feedback == "VALID_SOLUTION":
        return {"status": "SOLVED", "final_output": proposal}

    # CASO 2: STEP VALIDO (Avanzamento)
    if feedback == "VALID_STEP":
        # Check for stagnation: if the proposal is identical to the current partial solution,
        # it means the agent is not advancing.
        if proposal == partial:
             # Treat as invalid to force a change
             log_line = f"STAGNATION: Proposal [{proposal}] is identical to Partial Solution. You must EXTEND or CHANGE it."
             new_forbidden = (forbidden + "\n" + log_line).strip()
             return {"forbidden_moves": new_forbidden, "attempt_count": state["attempt_count"] + 1, "status": "SEARCHING"}

        return {"partial_solution": proposal, "attempt_count": 0, "manager_instruction": "", "status": "SEARCHING"}

    # CASO 3: NECESSITÀ DI BACKTRACK
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

    # CASO 4: ERRORE SEMPLICE (Riprova lo stesso step)
    # Append a human-readable forbidden note so the model can avoid repeating it
    log_line = f"INVALID_PROPOSAL at partial=[{partial}] -> proposal=[{proposal}] ; reason=[{feedback}]"
    new_forbidden = (forbidden + "\n" + log_line).strip()

    return {"forbidden_moves": new_forbidden, "attempt_count": state["attempt_count"] + 1, "status": "SEARCHING"}


def parser_node(state: GraphState) -> Dict[str, Any]:
    # Formatta l'output finale in base alle richieste del dataset
    if state["status"] == "FAILED":
        return {"final_output": "No solution"}

    raw = state["final_output"]
    if raw == "NO SOLUTION FOUND":
        return {"final_output": "No solution"}

    # The final output is produced/controlled by the model as a string.
    return {"final_output": raw}


# ==========================================
# 4. GRAFO E ROUTING
# ==========================================


def router(state: GraphState) -> str:
    if state["status"] == "SOLVED":
        return "parser"
    if state["status"] == "FAILED":
        return "parser"
    return "proposer"  # Continua il loop (avanti o indietro è gestito dallo stato)


def build_backtracking_solver():
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
# 5. ESECUZIONE DEMO
# ==========================================

if __name__ == "__main__":
    # Esempio Hamiltoniano
    # A-B, B-C, A-C. Percorso: A->B->C
    graphwiz_dataset = load_dataset("GraphWiz/GraphInstruct")

    example = graphwiz_dataset["train"][0]

    print("ESEMPIO QUERY:")
    print(example["query"])
    print("ESEMPIO TASK:")
    print(example["task"])
    print("ESEMPIO SOLUZIONE:")
    print(example["answer"])

    initial_state: GraphState = {
        "query": str(example["query"]),
        "task_type": str(example["task"]),
        "partial_solution": "",  # Inizia vuoto, il proposer suggerirà il primo nodo
        "forbidden_moves": "",
        "current_proposal": "",
        "verifier_feedback": "",
        "attempt_count": 0,
        "final_output": "",
        "status": "SEARCHING",
    }

    app = build_backtracking_solver()

    # Esecuzione con streaming per vedere i passaggi
    for step in app.stream(initial_state):
        print(step)
