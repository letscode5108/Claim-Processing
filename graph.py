from typing import TypedDict, Dict, List, Any
from langgraph.graph import StateGraph, END, START
from agents.segregator import run_segregator
from agents.id_agent import run_id_agent
from agents.discharge_agent import run_discharge_agent
from agents.bill_agent import run_bill_agent
from time import sleep



class ClaimState(TypedDict):
    pdf_bytes: bytes
    claim_id: str
    page_classification: Dict[str, List[int]]   # segregator output
    id_data: Dict[str, Any]                      # id agent output
    discharge_data: Dict[str, Any]               # discharge agent output
    bill_data: Dict[str, Any]                    # bill agent output
    final_result: Dict[str, Any]                 # aggregator output



def segregator_node(state: ClaimState) -> ClaimState:
    """Classify every page into a document type."""
    print(" Segregator: classifying pages...")
    classification = run_segregator(state["pdf_bytes"])
    print(f"   Result: {classification}")
    return {**state, "page_classification": classification}


def id_node(state: ClaimState) -> ClaimState:
    """Extract identity info from identity_document pages."""
    print("  ID Agent: extracting identity info...")
    page_nums = state["page_classification"].get("identity_document", [])
    result = run_id_agent(state["pdf_bytes"], page_nums)
    print(f"   Result: {result}")
    return {**state, "id_data": result}


def discharge_node(state: ClaimState) -> ClaimState:
    """Extract discharge summary info from discharge_summary pages."""
    print(" Discharge Agent: extracting discharge info...")
    # sleep(70) 
    page_nums = state["page_classification"].get("discharge_summary", [])
    result = run_discharge_agent(state["pdf_bytes"], page_nums)
    print(f"   Result: {result}")
    return {**state, "discharge_data": result}


def bill_node(state: ClaimState) -> ClaimState:
    """Extract itemized bill info from itemized_bill pages."""
    print(" Bill Agent: extracting bill info...")
    # sleep(70)
    page_nums = state["page_classification"].get("itemized_bill", [])
    result = run_bill_agent(state["pdf_bytes"], page_nums)
    print(f"   Result: {result}")
    return {**state, "bill_data": result}


def aggregator_node(state: ClaimState) -> ClaimState:
    """Combine all agent results into one final JSON."""
    print(" Aggregator: combining results...")
    final_result = {
        "claim_id": state["claim_id"],
        "page_classification": state["page_classification"],
        "extracted_data": {
            "identity":          state.get("id_data", {}),
            "discharge_summary": state.get("discharge_data", {}),
            "itemized_bill":     state.get("bill_data", {}),
        }
    }
    return {**state, "final_result": final_result}



def build_graph():
    builder = StateGraph(ClaimState)

    # Add nodes
    builder.add_node("segregator", segregator_node)
    builder.add_node("id_agent",   id_node)
    builder.add_node("discharge_agent", discharge_node)
    builder.add_node("bill_agent", bill_node)
    builder.add_node("aggregator", aggregator_node)

    # Edges
    builder.add_edge(START, "segregator")



    builder.add_edge("segregator",      "id_agent")
    builder.add_edge("id_agent",        "discharge_agent")
    builder.add_edge("discharge_agent", "bill_agent")
    builder.add_edge("bill_agent",      "aggregator")

    # Aggregator ends the graph
    builder.add_edge("aggregator", END)

    return builder.compile()



claim_graph = build_graph()


















#    # Segregator fans out to all 3 agents in parallel
#     builder.add_edge("segregator", "id_agent")
#     builder.add_edge("segregator", "discharge_agent")
#     builder.add_edge("segregator", "bill_agent")

#     # All 3 agents feed into aggregator
#     builder.add_edge("id_agent",        "aggregator")
#     builder.add_edge("discharge_agent", "aggregator")
#     builder.add_edge("bill_agent",      "aggregator")