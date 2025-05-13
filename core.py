from openai import AzureOpenAI
import os
from web_agent import search_google  # Assuming you have a proper search function
from langgraph.graph import END
from pydantic import BaseModel
from typing import List, Tuple, Union
from typing_extensions import TypedDict
plan_step, execute_step, replan_step = "", "", ""
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI client setup
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
)

# Azure Chat wrapper
class AzureChatWrapper:
    def __init__(self, client, deployment_name):
        self.client = client
        self.deployment = deployment_name

    def invoke(self, inputs):
        messages = inputs["messages"]
        formatted = [{"role": role, "content": content} for role, content in messages]
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=formatted
        )
        return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

llm = AzureChatWrapper(client, deployment_name="gpt-4.1")

# Types
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: str

class Plan(BaseModel):
    steps: List[str]

class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan]

def rewrite_query(original_query: str) -> str:
    """Rewrite the search query for better results from the Google Search API."""
    print(f"Original query: {original_query}")
    rewrite_prompt = f"""
    You are an expert in optimizing search engine queries. Rewrite the following query to make it more specific, clear, and likely to yield relevant results:
    
    Original Query: "{original_query}"
    
    Provide a rewritten query that improves the search results to get accurately expected results on internet.
    Return only the rewritten query without any additional text.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specializing in search engine optimization."},
            {"role": "user", "content": rewrite_prompt}
        ]
    )
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

# Prompts
planner_prompt = """You are a research agent. For the given objective, come up with a simple step-by-step plan.  
This plan should involve individual tasks that, if executed correctly, will yield the correct answer.  
Do not add any unnecessary steps. Make sure that each step has all the information needed."""

# Planning step
def plan_step(state: PlanExecute):
    query = state["input"]
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
            {"role": "user", "content": f"{planner_prompt}\n\nObjective: {query}\n\nWeb search results: {search_google(rewrite_query(query))}\n\nInclude attributions to the sources you used in your plan with links to them."}
        ]
    )
    plan_text = response.choices[0].message.content
    steps = [step.strip() for step in plan_text.split("\n") if step.strip()]
    return {"plan": steps}

def research_task(task: str, plan: List[str]) -> str:
    """Perform the research for the given task."""
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    research_prompt = f"For the following plan:\n{plan_str}\n\nResearch the following step: {task}"
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant gathering information for a task."},
            {"role": "user", "content": research_prompt}
        ]
    )
    research_result = response.choices[0].message.content
    return research_result

def write_task(task: str, research_result: str) -> str:
    """Generate the final output for the task based on the research."""
    write_prompt = f"Based on the following research:\n{research_result}\n\nWrite a detailed and well-structured response for the task: {task}\n You should write it as a research section"
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant writing a detailed response based on research."},
            {"role": "user", "content": write_prompt}
        ]
    )
    write_result = response.choices[0].message.content
    return write_result

def execute_step(state: PlanExecute):
    """Execute a single step by performing research and writing."""
    plan = state["plan"]
    task = plan[0]

    # Perform research
    research_result = research_task(task, plan)

    # Generate the final output
    write_result = write_task(task, research_result)

    return {"past_steps": [(task, write_result)]}


# Replan step
def replan_step(state: PlanExecute):
    remaining_steps = state["plan"][1:]  # remove the executed one
    if not remaining_steps:
        # If no more steps, summarize and return
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
                {"role": "user", "content": f"Given the steps already taken:\n{state['past_steps']}\nGenerate the deep and detailed final research report.\n\nMake sure to include all the information.\n\nInclude attributions to all the sources you used with links to them."}
            ]
        )
        return {"response": response.choices[0].message.content}
    else:
        return {"plan": remaining_steps}

# Control flow
def should_end(state: PlanExecute):
    return END if "response" in state and state["response"] else "agent"