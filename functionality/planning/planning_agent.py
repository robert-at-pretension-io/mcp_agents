from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from agents import Agent, RunContextWrapper
from functionality.planning.planning_context import PlanningContext

# Define structured models for the planning agent
class ToolParameter(BaseModel):
    """Parameter for a tool call"""
    name: str = Field(description="Name of the parameter")
    value: str = Field(description="Value of the parameter")
    description: Optional[str] = Field(None, description="Description of what this parameter does")

class PlanStep(BaseModel):
    """A step in the execution plan"""
    agent_name: str = Field(description="Name of the agent to use")
    tool_name: Optional[str] = Field(None, description="Name of the tool to call")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Parameters for the tool call")
    description: str = Field(description="What this step accomplishes")
    expected_output: str = Field(description="What output is expected from this step")

class Contingency(BaseModel):
    """A contingency plan for potential failures"""
    condition: str = Field(description="Condition that triggers this contingency")
    steps: List[PlanStep] = Field(description="Steps to take in this contingency")
    description: str = Field(description="Description of this contingency plan")

# Define the main output structure for the planning agent
class ExecutionPlan(BaseModel):
    """A proposed execution plan using available tools and agents"""
    goal: str = Field(description="The overall goal of this plan")
    steps: List[PlanStep] = Field(description="Sequence of tool/agent calls to make")
    contingencies: List[Contingency] = Field(default_factory=list, description="Plans for handling potential failures")
    rationale: str = Field(description="Reasoning behind this plan")

# Define dynamic instructions function
def planning_instructions(ctx: RunContextWrapper[PlanningContext], agent: Agent) -> str:
    """
    Dynamic instructions for planning agent that includes information about
    all available agents and tools from the registry.
    """
    # Get agent registry from the context
    registry = getattr(ctx.context, 'agent_registry', {})
    
    # Format available agents and their tools
    agent_descriptions = []
    
    for agent_name, agent_info in registry.items():
        # Skip the planning agent itself to avoid circular references
        if agent_name == "planner":
            continue
            
        agent_desc = f"""
## {agent_name}
{agent_info.get('description', 'No description available')}

### Tools:
"""
        
        # Add tool details if available
        tools = agent_info.get('tools', [])
        if tools:
            for tool in tools:
                tool_name = tool.get('name', 'Unknown')
                tool_desc = tool.get('description', 'No description available')
                
                # Start with basic tool info
                agent_desc += f"- **{tool_name}**: {tool_desc}\n"
                
                # Add parameter information if available
                parameters = tool.get('parameters', [])
                if parameters:
                    agent_desc += "  Parameters:\n"
                    for param in parameters:
                        param_name = param.get('name', 'unnamed')
                        param_type = param.get('type', 'unknown')
                        param_desc = param.get('description', '')
                        required = param.get('required', False)
                        req_text = "required" if required else "optional"
                        agent_desc += f"    - {param_name} ({param_type}, {req_text}): {param_desc}\n"
        else:
            agent_desc += "- No tools available\n"
        
        agent_descriptions.append(agent_desc)
    
    # Join all agent descriptions
    all_agents_info = "\n".join(agent_descriptions) if agent_descriptions else "No agents available"
    
    # Get permissions
    try:
        permissions = ", ".join(ctx.context.get_permissions())
    except Exception:
        permissions = "read, write"
    
    # Return full instructions with agent and tool information
    return f"""You are a Planning Agent that creates detailed execution plans using available tools and agents.

# Available Agents and Their Tools
{all_agents_info}

# Your Task
Your job is to:
1. Analyze the user's request carefully
2. Break down the request into a sequence of steps
3. For each step, identify the most appropriate agent and tool to use
4. Create a comprehensive execution plan that achieves the user's goal
5. Include contingency plans for potential failures

# Important Guidelines
- Be thorough in your planning
- Consider dependencies between steps
- Optimize for efficiency and reliability
- Provide clear rationale for your choices
- Consider the appropriate order of operations

# Output Structure
Your response must be structured as an ExecutionPlan with:
- goal: A clear statement of what the plan aims to achieve
- steps: A list of PlanStep objects, each containing:
  - agent_name: The name of the agent to use (e.g., "shell", "search")
  - tool_name: The specific tool to use (if applicable)
  - parameters: A list of ToolParameter objects with:
    - name: Parameter name
    - value: Parameter value
    - description: What this parameter does
  - description: What this step accomplishes
  - expected_output: What output is expected from this step
- contingencies: A list of Contingency objects for handling failures, each containing:
  - condition: When this contingency should be triggered
  - steps: Alternative steps to take (same structure as main steps)
  - description: What this contingency addresses
- rationale: Your reasoning for the plan structure

User permissions: {permissions}
Session ID: {ctx.context.session_id}

Important: Return your response as a structured ExecutionPlan object, not as freeform text.
"""

# Create the planning agent with dynamic instructions
planning_agent = Agent[PlanningContext](
    name="Planning Agent",
    instructions=planning_instructions,
    model="gpt-4o",
    output_type=ExecutionPlan
)