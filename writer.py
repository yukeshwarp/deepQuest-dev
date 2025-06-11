from config import client


def report_writer(context):
    """Generates a highly detailed research report from completed steps and results, with full source attribution and comprehensive coverage."""
    report_prompt = (
        f"Given the following completed research steps and their results:\n{context}\n\n"
        "As an autonomous research agent, write a highly detailed, exhaustive, and well-structured research report that answers the original query. "
        "Include attribution to all sources referenced or used in any step. "
        "Ensure that every piece of information, even if only slightly related to the research topic, is included and clearly explained. "
        "Organize the report with clear sections, provide in-depth analysis, and cite all sources explicitly. "
        "If possible, include a bibliography or references section at the end listing all sources."
    )
    report_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are a research report writing assistant.",
            },
            {"role": "user", "content": report_prompt},
        ],
    )
    return report_response.choices[0].message.content

# Feedback loop