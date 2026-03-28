"""
Autonomous Research Agent - Assignment 2
=========================================

An autonomous research agent built with LangChain that searches the web,
gathers knowledge from multiple sources, analyzes data, and generates
a structured research report on any user-specified topic.

Technologies Used:
    - LangChain (ReAct Agent with AgentExecutor)
    - ChatGroq (llama3-70b-8192)
    - Tavily Search (web search)
    - Wikipedia (encyclopedic knowledge)

Author: Aryan Raj
"""

import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from tools import get_search_tools

# Load environment variables from .env file
load_dotenv()


def initialize_llm():
    """
    Initialize the ChatGroq LLM with llama3-70b-8192 model.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=4096,
        api_key=groq_api_key,
    )
    return llm


def create_agent_prompt():
    """
    Create the ReAct prompt template for the research agent.
    
    Returns:
        PromptTemplate: The prompt template for the ReAct agent
    """
    template = """You are an Autonomous Research Agent. Your task is to conduct 
thorough research on a given topic and produce a comprehensive, well-structured 
research report.

Your research process should follow these steps:
1. Search the web using Tavily to find current and relevant information
2. Query Wikipedia for background knowledge and foundational concepts
3. Cross-reference and analyze the gathered information
4. Synthesize insights from multiple sources
5. Generate a structured research report

IMPORTANT GUIDELINES:
- Use BOTH the tavily_search and wikipedia tools to gather information
- Make multiple searches with different queries to get comprehensive coverage
- Focus on accuracy and cite factual information
- Provide balanced perspectives on the topic
- Ensure the report is detailed and academically structured

You have access to the following tools:

{tools}

Use the following format:

Question: the input question/topic you must research
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to compile a comprehensive report
Final Answer: A structured research report with the following sections:

# Research Report: [Topic Name]

## 1. Introduction
[Provide a comprehensive introduction to the topic, including its definition, 
background, and significance]

## 2. Key Insights
[Present the most important findings from your research, organized as detailed 
bullet points with explanations]

## 3. Applications
[Discuss real-world applications and use cases of the topic]

## 4. Advantages
[List and explain the key advantages or benefits]

## 5. Challenges
[Discuss current challenges, limitations, and areas of concern]

## 6. Conclusion
[Provide a thoughtful summary of the research findings and future outlook]

Begin your research now!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "",
            "tool_names": "",
        },
    )
    return prompt


def build_agent():
    """
    Build the ReAct research agent with tools and LLM.
    
    Returns:
        AgentExecutor: The configured agent executor
    """
    # Initialize components
    llm = initialize_llm()
    tools = get_search_tools()
    prompt = create_agent_prompt()

    # Update prompt with actual tool information
    tool_strings = "\n".join(
        [f"{tool.name}: {tool.description}" for tool in tools]
    )
    tool_names = ", ".join([tool.name for tool in tools])
    prompt = prompt.partial(tools=tool_strings, tool_names=tool_names)

    # Create ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # Wrap in AgentExecutor for execution management
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=True,
    )

    return agent_executor


def save_report(topic, report):
    """
    Save the generated research report to the outputs folder.
    
    Args:
        topic (str): The research topic
        report (str): The generated report content
    
    Returns:
        str: The path to the saved report file
    """
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Sanitize topic for filename
    safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()
    safe_topic = re.sub(r'[\s]+', '_', safe_topic).lower()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{safe_topic}_{timestamp}.txt"
    filepath = os.path.join("outputs", filename)

    # Write report to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"AUTONOMOUS RESEARCH AGENT - RESEARCH REPORT\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: ChatGroq (llama3-70b-8192)\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(report)
        f.write(f"\n\n{'=' * 60}\n")
        f.write(f"END OF REPORT\n")
        f.write(f"{'=' * 60}\n")

    return filepath


def display_banner():
    """Display the application banner."""
    print("\n" + "=" * 60)
    print("  🔬 AUTONOMOUS RESEARCH AGENT")
    print("  Assignment 2 - LangChain + ChatGroq + ReAct Agent")
    print("=" * 60)
    print("  Model   : llama3-70b-8192 (Groq)")
    print("  Tools   : Tavily Search, Wikipedia")
    print("  Strategy: ReAct (Reasoning + Acting)")
    print("=" * 60 + "\n")


def main():
    """Main function to run the Autonomous Research Agent."""
    display_banner()

    # Step 1: Ask user for research topic
    topic = input("📝 Enter your research topic: ").strip()

    if not topic:
        print("❌ No topic provided. Exiting.")
        return

    print(f"\n🔍 Starting research on: '{topic}'")
    print("⏳ The agent will search the web and Wikipedia...\n")
    print("-" * 60)

    try:
        # Step 2-5: Build and run the agent
        agent_executor = build_agent()

        result = agent_executor.invoke(
            {"input": f"Conduct comprehensive research on: {topic}"}
        )

        # Step 6: Extract and display the report
        report = result.get("output", "No report generated.")

        print("\n" + "=" * 60)
        print("📄 RESEARCH REPORT")
        print("=" * 60 + "\n")
        print(report)

        # Step 7: Save report to outputs folder
        filepath = save_report(topic, report)
        print(f"\n✅ Report saved to: {filepath}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your API keys and internet connection.")
        raise


if __name__ == "__main__":
    main()
