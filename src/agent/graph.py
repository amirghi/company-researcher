import asyncio
from typing import cast, Any, Literal
import json
import os

import openai
from openai import OpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
#from fastapi import FastAPI

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)
from tavily import AsyncTavilyClient  # Add this import

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)

def gpt_model(prompt, model="gpt-4o-mini", temperature=0):

#     response = """
# {
#   "company_name": "Intuit Inc.",
#   "founding_year": 1983,
#   "founder_names": [
#     "Scott Cook",
#     "Tom Proulx"
#   ],
#   "product_description": "Intuit Inc. is a leading financial software company that provides a range of products and services aimed at simplifying bookkeeping and tax preparation for individuals and small businesses, including TurboTax, QuickBooks, Mint, and Credit Karma.",
#   "funding_summary": "Intuit has a history of raising funds through various rounds, with its latest funding being a Post-IPO Debt round on September 12, 2023. The company has raised funding over three rounds from multiple investors, including private equity and venture capital firms. Notably, Intuit went public in 1993 and has since maintained a strong market presence."
# }
# """
#     print("Skipping API call")
#     return response

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
        max_tokens=1000
    )
    print(f"\nPrompt: <{prompt}>\nResponse: <\n{response.choices[0].message.content}>\n")
    return response.choices[0].message.content

# Search

tavily_api_key = os.getenv("TAVILY_API_KEY")
print(tavily_api_key)
tavily_async_client = AsyncTavilyClient(api_key=tavily_api_key)


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    prompt = f"System: {query_instructions}\nUser: Please generate a list of search queries related to the schema that you want to populate."
    response_text = gpt_model(prompt)
    results = json.loads(response_text)

    # Queries
    query_list = [query for query in results['queries']]
    return {"search_queries": query_list}


async def research_company(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Search tasks
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    response_text = gpt_model(p)
    return {"completed_notes": [response_text]}


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    response_text = gpt_model(system_prompt)
    result = json.loads(response_text)
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    response_text = gpt_model(system_prompt)
    result = json.loads(response_text)

    if result['is_satisfactory']:
        return {"is_satisfactory": result['is_satisfactory']}
    else:
        return {
            "is_satisfactory": result['is_satisfactory'],
            "search_queries": result['search_queries'],
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    # If we've exceeded max steps, end even if not satisfactory
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()
