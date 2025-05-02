from openai import AzureOpenAI
import os
from web_agent import search_bing  # Assuming you have a proper search function
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Tuple, Union
from typing_extensions import TypedDict
import streamlit as st
import operator
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


# Prompts
planner_prompt = """You are a finance research agent working in Oct 2024. For the given objective, come up with a simple step-by-step plan. 
This plan should involve individual tasks that, if executed correctly, will yield the correct answer. 
Do not add any superfluous steps. Make sure that each step has all the information needed."""

# Planning step
def plan_step(state: PlanExecute):
    query = state["input"]
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
            {"role": "user", "content": f"{planner_prompt}\n\nObjective: {query}\n\nWeb search results: {search_bing(query)}\n\nInclude attributions to the sources you used in your plan with links to them."}
        ]
    )
    plan_text = response.choices[0].message.content
    steps = [step.strip() for step in plan_text.split("\n") if step.strip()]
    return {"plan": steps}

def execute_step(state: PlanExecute):
    plan = state["plan"]
    task = plan[0]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task_prompt = f"For the following plan:\n{plan_str}\n\nYou are tasked with executing step: {task}"
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
            {"role": "user", "content": task_prompt}
        ]
    )
    result = response.choices[0].message.content
    return {"past_steps": [(task, result)]}


# Replan step
def replan_step(state: PlanExecute):
    remaining_steps = state["plan"][1:]  # remove the executed one
    if not remaining_steps:
        # If no more steps, summarize and return
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
                {"role": "user", "content": f"Given the steps already taken:\n{state['past_steps']}\nGenerate the deep and detailed final research report.\n\nMake sure to include all the information from the previous steps.\n\nInclude attributions to the sources you used in your plan with links to them."}
            ]
        )
        return {"response": response.choices[0].message.content}
    else:
        return {"plan": remaining_steps}

# Control flow
def should_end(state: PlanExecute):
    return END if "response" in state and state["response"] else "agent"