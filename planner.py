import streamlit as st
from web_agent import search_google
from dotenv import load_dotenv
from config import client
import logging

load_dotenv()


def plan_research(query, max_steps=20):
    """Ask the LLM to generate a step-by-step research plan for the query, with a dynamic max_steps limit."""
    plan_prompt = (
        "You are an expert research agent. "
        f"Given the following user query, create a clear, step-by-step research plan. "
        f"Each step should be actionable and focused on gathering or synthesizing information needed to answer the query. "
        f"Do not add unnecessary steps. Return the plan as a numbered list. "
        f"Do not exceed {max_steps} steps in your plan.\n\n"
        f"User Query: {query}"
    )
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a research planning assistant."},
            {"role": "user", "content": plan_prompt},
        ],
    )
    plan_text = response.choices[0].message.content
    steps = [
        step[2:].strip()
        for step in plan_text.split("\n")
        if step.strip() and step[0].isdigit()
    ]
    return steps

def replanner(context, steps, replan_rounds, max_replan_rounds, replan_limit_reached, max_steps=20):
    """Handles replanning logic and returns updated steps, replan_rounds, and replan_limit_reached, with a dynamic max_steps limit."""
    if replan_limit_reached:
        return steps, replan_rounds, replan_limit_reached

    replan_prompt = (
        f"Given the completed steps and results so far:\n{context}\n\n"
        f"As an autonomous agent, do you need to add any new steps to fully answer the original query? "
        f"If yes, list them as a numbered list, but do not exceed a total of {max_steps} steps in the plan (including already completed and planned steps). "
        f"If not, reply 'No additional steps needed.'"
    )
    replan_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a research planning assistant."},
            {"role": "user", "content": replan_prompt},
        ],
    )
    replan_text = replan_response.choices[0].message.content.strip().lower()
    if "no additional steps needed" in replan_text:
        replan_rounds = 0  # Reset replan rounds if no new steps
        return steps, replan_rounds, replan_limit_reached

    # Parse new steps, avoid duplicates, and enforce max_steps
    new_steps = [
        s[2:].strip() for s in replan_text.split("\n") if s.strip() and s[0].isdigit()
    ]
    # Only add steps if total does not exceed max_steps
    allowed_new_steps = new_steps[: max(0, max_steps - len(steps))]
    new_unique_steps = [new_step for new_step in allowed_new_steps if new_step not in steps]
    if new_unique_steps:
        steps.extend(new_unique_steps)
        replan_rounds += 1
        if replan_rounds > max_replan_rounds:
            logging.info(
                "Maximum replanning rounds reached. Will finish executing current plan and stop replanning."
            )
            replan_limit_reached = True
    else:
        replan_rounds += 1
        if replan_rounds > max_replan_rounds:
            logging.info(
                "Maximum replanning rounds reached (no new unique steps). Will finish executing current plan and stop replanning."
            )
            replan_limit_reached = True
    return steps, replan_rounds, replan_limit_reached