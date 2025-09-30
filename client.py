
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

server_params = StdioServerParameters(
    command="python3",
    # Full absolute path to the math_server.py file
    args=["math_server.py"],
)

import asyncio

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            print("Initialized connection")
            # Get tools
            tools = await load_mcp_tools(session)
            print("Loaded tools")
            # Create and run the agent
            agent = create_react_agent("openai:gpt-4.1", tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response["messages"][-1].content)
            
if __name__ == "__main__":
    asyncio.run(main())