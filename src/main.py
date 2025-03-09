#!/usr/bin/env python3
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import init, Fore, Style
import argparse
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math  # Add this import to check for NaN
#from typing import HumanMessage
from langchain_core.messages import HumanMessage


#from parse import parse_hedge_fund_response
from utils.progress import progress


# Import agent modules
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from agents.valuation import valuation_agent

# Import state and utility functions
from graph.state import AgentState
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
from utils.visualize import save_graph_as_png

# Load environment variables and initialize colorama
load_dotenv()
init(autoreset=True)

def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None

# Add this new function to recursively remove "NaN" string values
def remove_nan_values(data):
    if isinstance(data, dict):
        return {
            k: remove_nan_values(v) 
            for k, v in data.items() 
            if not (isinstance(v, float) and math.isnan(v)) 
            and not (isinstance(v, str) and v.strip().lower() == "nan")
        }
    elif isinstance(data, list):
        return [
            remove_nan_values(item) 
            for item in data 
            if not (isinstance(item, float) and math.isnan(item)) 
            and not (isinstance(item, str) and item.strip().lower() == "nan")
        ]
    else:
        return data


"""
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o-mini",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()
    try:
        # Always create a workflow using the provided (all) analysts
        workflow = create_workflow(selected_analysts)
        agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data."
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        # Parse decisions and analyst signals
        decisions = parse_hedge_fund_response(final_state["messages"][-1].content)
        analyst_signals = final_state["data"]["analyst_signals"]

        # Filter out NaN values from decisions (assuming it's a dict of ticker: value pairs)
        if decisions is not None:
            decisions = {k: v for k, v in decisions.items() if not (isinstance(v, float) and math.isnan(v))}
        # Filter out NaN values from analyst_signals
        analyst_signals = {k: v for k, v in analyst_signals.items() if not (isinstance(v, float) and math.isnan(v))}

        return {
            "decisions": decisions,
            "analyst_signals": analyst_signals,
        }
    finally:
        # Stop progress tracking
        progress.stop()
"""

def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts.
    
    In non-interactive mode, if no analysts are provided, all available agents are used.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Retrieve all available analyst nodes
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none specified
    if selected_analysts is None or not selected_analysts:
        selected_analysts = list(analyst_nodes.keys())

    # Add each selected analyst node to the workflow
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management nodes
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect each analyst's output to the risk management agent
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o-mini",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()
    try:
        # Always create a workflow using the provided (all) analysts
        workflow = create_workflow(selected_analysts)
        agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data."
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        # Parse decisions and analyst signals
        decisions = parse_hedge_fund_response(final_state["messages"][-1].content)
        analyst_signals = final_state["data"]["analyst_signals"]

        # Apply the recursive NaN removal to both decisions and analyst_signals
        if decisions is not None:
            decisions = remove_nan_values(decisions)
        analyst_signals = remove_nan_values(analyst_signals)

        # Original filtering for float NaN values (optional, kept for robustness)
        if decisions is not None:
            decisions = {k: v for k, v in decisions.items() if not (isinstance(v, float) and math.isnan(v))}
        analyst_signals = {k: v for k, v in analyst_signals.items() if not (isinstance(v, float) and math.isnan(v))}

        return {
            "decisions": decisions,
            "analyst_signals": analyst_signals,
        }
    finally:
        # Stop progress tracking
        progress.stop()
def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state

def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts.
    
    In non-interactive mode, if no analysts are provided, all available agents are used.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Retrieve all available analyst nodes
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none specified
    if selected_analysts is None or not selected_analysts:
        selected_analysts = list(analyst_nodes.keys())

    # Add each selected analyst node to the workflow
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management nodes
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect each analyst's output to the risk management agent
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position. Defaults to 100000.0"
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement. Defaults to 0.0"
    )
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    args = parser.parse_args()

    # Parse tickers from the comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Non-interactive mode: always run all available analyst agents
    selected_analysts = list(get_analyst_nodes().keys())

    # Default model is -openai-4omini
    model_choice = "gpt-4o-mini"
    model_info = get_model_info(model_choice)
    if model_info:
        model_provider = model_info.provider.value
    else:
        model_provider = "OpenAI"

    # Create the workflow with selected analysts and compile it
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = "_".join(selected_analysts) + "_graph.png"
        save_graph_as_png(app, file_path)
        print(f"Agent graph saved to {file_path}")

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")
    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates (default: start date is 3 months before end date)
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize the portfolio with cash, margin, and positions for each ticker
    portfolio = {
        "cash": args.initial_cash,
        "margin_requirement": args.margin_requirement,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            } for ticker in tickers
        }
    }

    # Run the hedge fund trading system
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_choice,
        model_provider=model_provider,
    )

    print_trading_output(result)
