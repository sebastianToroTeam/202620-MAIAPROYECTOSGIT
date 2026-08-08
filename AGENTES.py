from langchain_ollama import ChatOllama 
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage 
from langchain.agents import create_agent 

# Define the mathematical tool the agent can use
@tool 
def calculate(a: float, b: float, operation: str) -> float: 
    """Performs a mathematical operation between two numbers. 
    The operation can be: add, subtract, multiply, or divide.
    """ 
    operations = {
        "add": a + b, 
        "subtract": a - b, 
        "multiply": a * b, 
        "divide": a / b if b != 0 else 0.0  # Safeguard to prevent division by zero
    } 
    return operations.get(operation, 0.0) 

# Initialize the local Ollama model (using Qwen 2.5 3B)
model = ChatOllama(model="qwen2.5:3b") 

# Create the ReAct agent graph using LangGraph's prebuilt function
# This automatically handles the loop between the model and the tools
agent = create_agent(model, tools=[calculate]) 

# Invoke the agent by passing a list of messages (following LangGraph's state structure)
result = agent.invoke(
    {"messages": [HumanMessage(content="How much is 127 multiplied by 48?")]} 
) 

# Print the content of the very last message in the conversation (the agent's final answer)
print(result["messages"][-1].content)
