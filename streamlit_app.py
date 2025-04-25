import streamlit as st
from web_agent import search_bing
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-01-preview",
)

st.title("deepQuest")

# Collect research inputs
research_topic = st.text_input("Research Topic", "Enter your research topic here")
description = st.text_area("Description", "Enter a description of your research topic here")
# keywords = st.text_input("Keywords", "Enter keywords related to your research topic here")
# research_question = st.text_area("Research Question", "Enter your research question here")
# research_methodology = st.text_area("Research Methodology", "Enter your research methodology here")

# Process when topic is submitted
if st.button("Generate Report") and research_topic:
    st.subheader("Step 1: Planning Report Structure")
    planning_prompt = f"""
    Based on the following inputs:
    - Topic: {research_topic}
    - Description: {description}

    Propose a report structure with an ordered list of logical sections. Each section should build on the previous one, culminating in a synthesized conclusion.
    """

    plan_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert research planner."},
            {"role": "user", "content": planning_prompt},
        ]
    )
    section_plan = plan_response.choices[0].message.content
    st.markdown(f"**Planned Sections:**\n{section_plan}")

    st.subheader("Step 2: Dynamic Section Synthesis and Writing")
    sections = [line.strip("- ").strip() for line in section_plan.strip().split("\n") if line.strip()]
    final_report = ""
    context_memory = ""

    for idx, section in enumerate(sections):
        st.markdown(f"### {section}")

        search_query = f"{section} {research_topic}"
        search_results = search_bing(search_query)

        # Fixing TypeError by ensuring proper result parsing
        if isinstance(search_results, list) and all(isinstance(result, dict) for result in search_results):
            research_context = "\n".join([result.get('snippet', '') for result in search_results[:5]])
        else:
            research_context = "No valid search results found."

        dynamic_prompt = f"""
        You are a research assistant. Use the context below and the ongoing research to write the section titled "{section}".

        Previous Context:
        {context_memory}

        New Web Context:
        {research_context}
        """

        section_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
                {"role": "user", "content": dynamic_prompt},
            ]
        )

        section_content = section_response.choices[0].message.content
        st.markdown(section_content)

        final_report += f"\n\n## {section}\n{section_content}"
        context_memory += f"\n\n{section_content}"

    st.subheader("Final Compiled Report")
    st.markdown(final_report)
    st.download_button("Download Report", data=final_report, file_name="research_report.md")
