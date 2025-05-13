from langgraph.graph import StateGraph, START, END
import streamlit as st
from dotenv import load_dotenv
from core import plan_step, execute_step, replan_step, should_end, PlanExecute
import streamlit as st
from docx import Document
from config import client
from io import BytesIO
from cache import create_redis_index, knn_search, generate_embedding
load_dotenv()

st.title("Agentic Research Assistant")
st.header("Research Report", divider="orange")

if "plan" not in st.session_state:
    st.session_state["plan"] = []

if "past_steps" not in st.session_state:
    st.session_state["past_steps"] = []

if "current_step_index" not in st.session_state:
    st.session_state["current_step_index"] = 0

if "response" not in st.session_state:
    st.session_state["response"] = ""
if "topic" not in st.session_state:
    st.session_state["topic"] = ""

create_redis_index()

def generate_docx(content: str) -> BytesIO:
    doc = Document()
    doc.add_paragraph(content)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Modified wrapper functions to properly update session state
def wrapped_plan_step(state: dict) -> dict:
    """Wrapper around plan_step to update session state"""
    result = plan_step(state)
    if "plan" in result:
        st.session_state["plan"] = result["plan"]
    return result

def wrapped_execute_step(state: dict) -> dict:
    """Wrapper around execute_step to update session state"""
    result = execute_step(state)
    # Make sure we preserve the full plan in state
    if "plan" not in result and "plan" in state:
        result["plan"] = state["plan"]
    
    # Update past steps
    if "past_steps" in result:
        current_past_steps = st.session_state.get("past_steps", [])
        new_steps = result.get("past_steps", [])
        st.session_state["past_steps"] = current_past_steps + new_steps
    
    return result

def wrapped_replan_step(state: dict) -> dict:
    """Wrapper around replan_step to update session state"""
    # Create a copy of the state to modify before passing to replan_step
    replan_state = state.copy()
    
    # Mark current step as completed by advancing the index
    st.session_state["current_step_index"] += 1
    
    # Only remove the first plan item in our modified state, preserving the original
    if "plan" in replan_state and replan_state["plan"]:
        replan_state["plan"] = replan_state["plan"][1:]
    
    # Call the original replan_step
    result = replan_step(replan_state)
    
    # Update session state with new plan if provided
    if "plan" in result:
        st.session_state["plan"] = result["plan"]
    
    return result

# Workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", wrapped_plan_step)
workflow.add_node("agent", wrapped_execute_step)
workflow.add_node("replan", wrapped_replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])
app = workflow.compile()

# Inputs
inputs = {
    "input": "",
    "plan": [],
    "past_steps": [],
    "response": ""
}

def run_sync_app(Topic: str):
    # Reset session state for a new run
    st.session_state["plan"] = []
    st.session_state["past_steps"] = []
    st.session_state["current_step_index"] = 0
    
    state = {
        "input": Topic,
        "plan": [],
        "past_steps": [],
        "response": ""
    }
    
    planner_prompt = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Describe the qualities and domain expertise of a research assistant best suited for the following research topic:\n\n{Topic}\n\nRespond only with a short paragraph."}
        ]
    ).choices[0].message.content

    current_node = START
    result = {}

    while current_node != END:
        if current_node == START:
            current_node = "planner"
            result = wrapped_plan_step(state)
            
        elif current_node == "planner":
            state.update(result)
            current_node = "agent"
            result = wrapped_execute_step(state)
            
        elif current_node == "agent":
            # Update state with result from agent
            state.update(result)
            # Do NOT remove items from the plan here - that's handled in wrapped_replan_step
            current_node = "replan"
            result = wrapped_replan_step(state)
            
        elif current_node == "replan":
            state.update(result)
            current_node = should_end(state)

    # st.write("\n✅ Final Output:\n")
    st.session_state["response"] = state["response"]
    # st.write(state["response"])
    # Perform KNN search for the final topic
    query_embedding = generate_embedding(Topic)
    similar_items = knn_search(query_embedding)
    # st.subheader("🔍 Similar Topics Found:")
    # for item_key, score in similar_items:
    #     st.markdown(f"- **Key:** {item_key}, **Similarity Score:** {score:.4f}")

st.write(st.session_state["response"])

def understand_intent(topic):
    return client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Understand the intent behind the following research topic:\n\n{topic}\n\nRespond only with a short paragraph."}
        ]
    ).choices[0].message.content

with st.sidebar:
    
    st.session_state["topic"] = understand_intent(st.text_area("Research Topic"))
    # st.header("📋 Current Research Plan")
    # if st.session_state["plan"]:
    #     for idx, step in enumerate(st.session_state["plan"], 1):
    #         st.markdown(f"**Step {idx}:** {step}")
    # else:
    #     st.write("Plan will appear here after planning.")
    if st.button("Run Research"):
        if st.session_state["topic"].strip():  # Only run if there's valid input
            print("topic:", st.session_state["topic"])
            run_sync_app(st.session_state["topic"])
            
            st.rerun()
            # st.rerun()  # Rerun to refresh the app state
            
        else:
            st.warning("Please enter a research topic.")

    st.subheader("✅ Executed Steps")
    if st.session_state["past_steps"]:
        for idx, (task, result) in enumerate(st.session_state["past_steps"], 1):
            st.markdown(f"**Step {idx}:** {task}\n\n")
    else:
        st.write("No steps executed yet.")

    docx_file = generate_docx(st.session_state["response"])

    # Streamlit download button for .docx
    st.download_button(
        label="Download Report",
        data=docx_file,
        file_name=f"{st.session_state['topic']}_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
