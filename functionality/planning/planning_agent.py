

from agents import Agent, RunContextWrapper, function_tool
from functionality.planning.planning_context import PlanningContext


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
            
        # Get tags for this agent
        tags = agent_info.get('tags', [])
        tag_text = f"[{', '.join(tags)}]" if tags else ""
            
        agent_desc = f"""
## {agent_name} {tag_text}
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
    
    # Join all agent descriptions with a warning if no agents are available
    if agent_descriptions:
        all_agents_info = "\n".join(agent_descriptions)
    else:
        all_agents_info = "WARNING: No agents or tools are currently available. " \
                         "Your plan should focus on gathering more information or asking clarifying questions."
    
    # Get permissions
    try:
        permissions = ", ".join(ctx.context.get_permissions())
    except Exception:
        permissions = "read, write"
    
    # Return full instructions with agent and tool information
    return f"""You are a Planning Agent that creates practical execution plans using available tools and agents.

# Available Agents and Their Tools
{all_agents_info}

# Your Task
Your job is to:
1. Analyze the user's request carefully
2. Create a clear, step-by-step plan using ONLY the tools and agents listed above
3. Be specific about which agent and tool to use for each step
4. Include any important considerations or potential issues

# Important Guidelines
- ONLY recommend tools and agents that are explicitly listed above
- DO NOT invent or assume the existence of tools that aren't listed
- Be realistic about what each tool can accomplish
- When uncertain, be conservative in your recommendations
- When there isn't enough context you must make the plan simply to ask clarifying questions
- Format your plan as a clear, readable text in markdown format

# Example Format
Your response should look like this:

## Execution Plan
**Goal**: [Describe the goal]

**Steps**:
1. Use agent `A` with tool `B` to do X
2. Use agent `C` to D about Y
3. Use agent `E` to F from Z
(replace those with tools/agents)

**Considerations**:
- Important factor 1
- Potential issue 2

User permissions: {permissions}
Session ID: {ctx.context.session_id}

"""

# Create the planning agent with dynamic instructions
planning_agent = Agent[PlanningContext](
    name="Planning Agent",
    instructions=planning_instructions,
    model="gpt-4o",
    tools=[]  # Add the dummy tool here
)
