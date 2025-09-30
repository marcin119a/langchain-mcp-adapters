from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import pandas as pd
import asyncio

# Load environment variables
load_dotenv()

server_params = StdioServerParameters(
    command="python3",
    args=["math_server.py"],  # Twój MCP server
)


# =======================================

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Initialized connection")

            # MCP tools
            tools = await load_mcp_tools(session)

            # Tworzymy agenta
            agent = create_react_agent("openai:gpt-4.1", tools)

            # Przykładowe zapytanie
            agent_response = await agent.ainvoke({
                "messages": "Pokaż mi mieszkania w Warszawie w cenie 800000–1200000 PLN i podaj mi linki do nich"
            })

            print(agent_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())