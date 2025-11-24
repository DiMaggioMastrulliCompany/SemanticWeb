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

PROPOSER_TEMPLATES = {
    "hamilton": (
        "Goal: Find a path visiting every node exactly once.\n"
        "Current Solution: {partial}\n"
        "Past Forbidden Moves: {forbidden}\n"
        "Instruction: Output EXACTLY one of the following:\n"
        "  1) The UPDATED Partial Solution string (in any format you choose) representing the state after applying your proposed step;\n"
        "  2) The single token 'BACKTRACK' if you want to backtrack;\n"
        "  3) The single token 'FINISHED' if the solution is complete.\n"
        "Do NOT include any extra commentary. The system will accept your returned string as the new partial solution when VALID."
    ),
    "substructure": (
        "Goal: Map pattern nodes to graph nodes.\n"
        "Current Solution: {partial}\n"
        "Past Forbidden Mappings: {forbidden}\n"
        "Instruction: Output EXACTLY one of the following:\n"
        "  1) The UPDATED Mapping string representing the state after applying your proposed mapping;\n"
        "  2) The single token 'BACKTRACK' if you want to backtrack;\n"
        "  3) The single token 'FINISHED' if the mapping is complete.\n"
        "Do NOT include any extra commentary. The system will accept your returned string as the new partial solution when VALID."
    ),
}

VERIFIER_TEMPLATES = {
    "hamilton": (
        "You are a strict logic verifier for Hamiltonian-path style tasks.\n"
        "Input: the graph description, the CURRENT Partial Solution (as provided by the proposer), and the PROPOSAL (which may be an updated partial, or the tokens BACKTRACK/FINISHED).\n"
        "Task: Determine whether the PROPOSAL is logically valid relative to the graph and the CURRENT Partial Solution.\n"
        "If the PROPOSAL is an updated partial, validate that the transition from the CURRENT partial to the PROPOSAL follows graph edges, does not repeat nodes improperly, and preserves constraints.\n"
        "If the PROPOSAL is 'FINISHED', validate that the PROPOSAL indeed represents a complete valid solution.\n"
        "If the PROPOSAL is 'BACKTRACK', return 'VALID' to acknowledge the backtrack request.\n"
        "Output EXACTLY 'VALID' or 'INVALID: <reason>'."
    ),
    "substructure": (
        "You are a strict logic verifier for substructure mapping tasks.\n"
        "Input: the graph description, the CURRENT Mapping (as provided by the proposer), and the PROPOSAL (which may be an updated mapping, or the tokens BACKTRACK/FINISHED).\n"
        "Task: Determine whether the PROPOSAL is logically valid relative to the graph and the CURRENT Mapping.\n"
        "If the PROPOSAL is an updated mapping, validate that edges and uniqueness constraints are respected.\n"
        "If the PROPOSAL is 'FINISHED', validate that the mapping is complete and correct.\n"
        "If the PROPOSAL is 'BACKTRACK', return 'VALID' to acknowledge the backtrack request.\n"
        "Output EXACTLY 'VALID' or 'INVALID: <reason>'."
    ),
}

# ==========================================
# 3. IMPLEMENTAZIONE NODI
# ==========================================


def proposer_node(state: GraphState) -> Dict[str, Any]:
    task = state["task_type"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]
    manager_instruction = state.get("manager_instruction", "")

    template = PROPOSER_TEMPLATES.get(task, PROPOSER_TEMPLATES["hamilton"])
    user_prompt = template.format(partial=str(partial), forbidden=str(forbidden))

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
        return {"verifier_feedback": "BACKTRACK_REQUEST"}

    task = state["task_type"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]

    template = VERIFIER_TEMPLATES.get(task, VERIFIER_TEMPLATES["hamilton"])
    user_prompt = template.format()
    # Provide full context for the verifier to reason about the proposal
    user_prompt += f"\nCurrent Partial: {partial}\nProposal: {proposal}\nForbidden: {forbidden}"

    msg = [("system", f"You are a strict logic verifier. Graph Description:\n{state['query']}"), ("user", user_prompt)]

    response = client.invoke(msg)
    content = response.content

    feedback = str(content).strip()

    # Normalizzazione output
    # Normalizzazione output: accept exact VALID, or treat anything else as INVALID with reason
    if feedback.strip() == "VALID":
        return {"verifier_feedback": "VALID"}
    else:
        return {"verifier_feedback": feedback}


def manager_node(state: GraphState) -> Dict[str, Any]:
    """Il cervello del Backtracking: gestisce lo Stack."""
    proposal = state["current_proposal"]
    feedback = state["verifier_feedback"]
    partial = state["partial_solution"]
    forbidden = state["forbidden_moves"]

    # CASO 1: SUCCESSO TOTALE
    if proposal == "FINISHED" and feedback == "VALID":
        return {"status": "SOLVED", "final_output": proposal}

    # CASO 2: STEP VALIDO (Avanzamento)
    # Here we accept the proposer's returned string as the new partial state.
    if feedback == "VALID":
        return {"partial_solution": proposal, "attempt_count": 0, "manager_instruction": "", "status": "SEARCHING"}

    # CASO 3: NECESSITÀ DI BACKTRACK
    should_backtrack = (feedback == "BACKTRACK_REQUEST") or (state["attempt_count"] >= 3)

    if should_backtrack:
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
