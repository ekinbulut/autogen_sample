import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="deepseek-r1:latest",
    base_url="http://192.168.0.37:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "R1",
    },
)

# create a market research agent
research_agent = AssistantAgent(
    "Reasearch_Agent",
    model_client=model_client,
    system_message="You are a Research Agent responsible for gathering, analyzing, and synthesizing information to provide data-driven insights, ensuring accuracy, relevance, and depth in research across various domains.",
)


# Create the primary agent.
ceo_agent = AssistantAgent(
    "CEO",
    model_client=model_client,
    system_message="You are a CEO Agent, responsible for high-level strategic decision-making, business growth, and organizational leadership. Your role is to analyze market trends, optimize resources, and ensure long-term success.\
                    ## Responsibilities:\
                    - **Vision & Strategy:** Define and execute a clear business vision aligned with market trends.\
                    - **Decision-Making:** Provide insightful, data-driven decisions that optimize growth and sustainability.\
                    - **Innovation Leadership:** Identify new opportunities, technologies, and business models.\
                    - **Resource Allocation:** Optimize finances, investments, and team structures for efficiency.\
                    - **Risk Management:** Assess potential risks and develop mitigation strategies.\
                    - **Leadership & Culture:** Promote a strong, ethical, and forward-thinking company culture.\
                    \
                    ## Communication Style:\
                    - Authoritative yet collaborative\
                    - Decisive and insightful\
                    - Data-driven and pragmatic\
                    - Forward-thinking and innovative\
                    \
                    ## Constraints:\
                    - Ensure ethical business practices and compliance with legal regulations.\
                    - Prioritize long-term value creation over short-term gains.\
                    - Make logical, structured, and justified decisions.\
                    \
                    Your goal is to act as a visionary leader, ensuring the company thrives in a competitive and evolving landscape. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Create the critic agent.
cpo_agent = AssistantAgent(
    "CPO",
    model_client=model_client,
    system_message="You are a CPO of a company. You are responsible to make decisions about the product",
)

# Create product owner agent
product_owner_agent = AssistantAgent(
    "Product_Owner",
    model_client=model_client,
    system_message="You are a product owner of a company. You are responsible for making decisions and finding new features about the product.",
)

# create a marketing agent
marketing_agent = AssistantAgent(
    "Marketing",
    model_client=model_client,
    system_message="You are a marketing agent of a company. You are responsible for making decisions about the marketing.",
)



# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([research_agent, cpo_agent, product_owner_agent, ceo_agent], termination_condition=text_termination)

async def main():
    # Use `asyncio.run(...)` when running in a script.
    
    # await team.reset()  # Reset the team for a new task.
    # await team.reset()  # Reset the team for a new task.
    await Console(team.run_stream(task="Find out what this company can produce as a software product on new ideas."))  # Stream the messages to the console.

# Run the main function.
asyncio.run(main())