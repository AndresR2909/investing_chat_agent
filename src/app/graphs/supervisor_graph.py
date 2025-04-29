##tools
import os

from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, END
from langgraph.types import Command


from app.tools.tools import optimize_portfolio_kde_cvar
from app.tools.custon_retrival_tool import buscar_documentos_fecha
from app.tools.custom_yahoo_finance_tools import (
    YahooFinanceNewsSummaryTool,
    YahooFinanceNewsTool,
)
from app.utils.prompts import load_prompt
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=5)

yahoo_finance_tool = YahooFinanceNewsTool()  # YahooFinanceNewsSummaryTool()

## Create Agent Supervisor

members = [
    "researcher",
    "finance_news_searcher",
    "risk_evaluator",
    "retriever_vector_data_base",
]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


system_prompt = load_prompt("src/app/prompt/v1_supervisor_agent.txt")
finance_news_searcher_prompt = load_prompt(
    "src/app/prompt/v1_finance_news_searcher_agent.txt"
)
researcher_prompt = load_prompt("src/app/prompt/v1_researcher_prompt.txt")
retraival_vdb_prompt = load_prompt("src/app/prompt/v1_retraival_vdb.txt")
risk_evaluator_prompt = load_prompt("src/app/prompt/v1_risk_evaluator.txt")


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = ChatOpenAI(model="gpt-4o-mini")  # "gpt-4o"


class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


## Construct Graph

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=researcher_prompt,  # "You are a researcher. DO NOT do any math."
)


def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


finance_news_searcher_agent = create_react_agent(
    llm,
    tools=[yahoo_finance_tool],
    prompt=finance_news_searcher_prompt,  # "You are a financial search news. DO NOT do any math and portafolio evaluator. Only search and summary news",
)


def finance_news_searcher_node(state: State) -> Command[Literal["supervisor"]]:
    result = finance_news_searcher_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="finance_news_searcher"
                )
            ]
        },
        goto="supervisor",
    )


risk_evaluator_agent = create_react_agent(
    llm,
    tools=[optimize_portfolio_kde_cvar],
    prompt=risk_evaluator_prompt,  # "You are a portafolio evaluator según el perfil de riesgo. DO NOT do any math and reserach. Only calulate portafolio risk",
)


def risk_evaluator_node(state: State) -> Command[Literal["supervisor"]]:
    result = risk_evaluator_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="risk_evaluator"
                )
            ]
        },
        goto="supervisor",
    )


retriever_vdb_agent = create_react_agent(
    llm,
    tools=[buscar_documentos_fecha],
    prompt=retraival_vdb_prompt,
)


def retriever_vdb_node(state: State) -> Command[Literal["supervisor"]]:
    result = retriever_vdb_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content,
                    name="retriever_vector_data_base",
                )
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("finance_news_searcher", finance_news_searcher_node)
builder.add_node("risk_evaluator", risk_evaluator_node)
builder.add_node("retriever_vector_data_base", retriever_vdb_node)
graph = builder.compile()
