from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
import asyncio
from dataclasses import dataclass, field
import uuid
import os

from agents import (
    Agent, Runner as OpenAIRunner, trace, 
    handoff, RunContextWrapper, ModelSettings
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from base_context import BaseAgentContext

# TypeVar for generic context types
T = TypeVar('T', bound=BaseAgentContext)

@dataclass
class ConversationMemory:
    """
    Maintains conversation history across agents
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history"""
        self.messages.append({
            "role": "user",
            "content": message
        })
    
    def add_assistant_message(self, message: str, agent_name: str = "assistant") -> None:
        """Add an assistant message to the conversation history"""
        self.messages.append({
            "role": "assistant",
            "agent": agent_name,
            "content": message
        })
    
    def add_tool_call(self, tool_name: str, input_data: Any, output: Any) -> None:
        """Add a tool call to the conversation history"""
        self.messages.append({
            "role": "tool",
            "tool": tool_name,
            "input": input_data,
            "output": output
        })
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages"""
        return self.messages[-limit:] if self.messages else []
    
    def get_formatted_history(self, limit: int = 10) -> str:
        """Get formatted conversation history for context injection"""
        recent = self.get_recent_messages(limit)
        formatted = []
        
        for msg in recent:
            if msg["role"] == "user":
                formatted.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                agent_prefix = f"[{msg['agent']}] " if "agent" in msg else ""
                formatted.append(f"Assistant {agent_prefix}: {msg['content']}")
            elif msg["role"] == "tool":
                formatted.append(f"Tool ({msg['tool']}): Input: {msg['input']} → Output: {msg['output']}")
        
        return "\n".join(formatted)

    def to_openai_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Convert the conversation memory to OpenAI message format
        for use with the OpenAI Agents SDK Runner
        """
        recent = self.get_recent_messages(limit)
        openai_messages = []
        
        for msg in recent:
            if msg["role"] == "user":
                openai_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                openai_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
            elif msg["role"] == "tool":
                # Skip tool messages as they'll be handled by the SDK
                pass
                
        return openai_messages


@dataclass
class OrchestratorContext(BaseAgentContext):
    """
    Context for the orchestrator agent
    """
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    
    def __post_init__(self):
        """Initialize orchestrator-specific attributes"""
        super().__post_init__()
        # Add any orchestrator-specific attributes
        self.set_attribute("role", "orchestrator")
        
        # Set working directory if not already set
        if "working_directory" not in self.attributes:
            import os
            self.set_attribute("working_directory", os.getcwd())
    
    def get_permissions(self) -> List[str]:
        """
        Get permissions for the orchestrator context.
        Extend base permissions for orchestration operations.
        """
        base_permissions = super().get_permissions()
        return base_permissions + ["execute"]


@dataclass
class AgentRegistryEntry:
    """
    Represents an agent entry in the registry with routing criteria
    """
    name: str
    agent: Agent
    description: str
    context_factory: Optional[Callable[[], Any]] = None


class AgentRegistry:
    """
    Registry of available agents with their metadata
    """
    def __init__(self):
        self.agents: Dict[str, AgentRegistryEntry] = {}
    
    def register(
        self,
        name: str,
        agent: Agent,
        description: str,
        context_factory: Optional[Callable[[], Any]] = None
    ) -> None:
        """
        Register an agent with the registry.
        
        Args:
            name: Unique identifier for the agent
            agent: The agent instance
            description: Description of when to use this agent
            context_factory: Optional function to create a context object for this agent
        """
        if name in self.agents:
            raise ValueError(f"Agent with name '{name}' already registered")
        
        self.agents[name] = AgentRegistryEntry(
            name=name,
            agent=agent,
            description=description,
            context_factory=context_factory
        )
    
    def get_agent(self, name: str) -> AgentRegistryEntry:
        """
        Get an agent by name.
        
        Args:
            name: The agent identifier
            
        Returns:
            The agent registry entry
            
        Raises:
            KeyError: If agent doesn't exist
        """
        if name not in self.agents:
            raise KeyError(f"No agent registered with name '{name}'")
        return self.agents[name]
    
    def get_all_agents(self) -> List[AgentRegistryEntry]:
        """
        Get all registered agents.
        
        Returns:
            List of all registered agent entries
        """
        return list(self.agents.values())
    
    def get_routing_descriptions(self) -> List[str]:
        """
        Get a list of agent descriptions for routing.
        
        Returns:
            List of agent descriptions
        """
        return [f"- {entry.name}: {entry.description}" for entry in self.agents.values()]
        
    def get_registry_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get structured information about all registered agents and their tools.
        
        This is used by the planning agent to understand available capabilities.
        
        Returns:
            Dictionary mapping agent names to their metadata including tools
        """
        registry_info = {}
        
        for name, entry in self.agents.items():
            agent_info = {
                "description": entry.description,
                "tools": []
            }
            
            # Extract tool information if available
            if hasattr(entry.agent, 'tools') and entry.agent.tools:
                for tool in entry.agent.tools:
                    # Get basic tool info
                    tool_name = getattr(tool, 'name', None)
                    
                    # Skip tools without names
                    if not tool_name:
                        continue
                        
                    # Get tool description
                    tool_description = getattr(tool, 'description', 'No description available')
                    
                    # Get parameter schema if available
                    parameters = []
                    if hasattr(tool, 'params_json_schema'):
                        schema = tool.params_json_schema
                        if isinstance(schema, dict) and 'properties' in schema:
                            for param_name, param_info in schema['properties'].items():
                                param_type = param_info.get('type', 'unknown')
                                param_desc = param_info.get('description', f'Parameter of type {param_type}')
                                parameters.append({
                                    "name": param_name,
                                    "type": param_type,
                                    "description": param_desc,
                                    "required": param_name in schema.get('required', [])
                                })
                                
                    # Create complete tool info
                    tool_info = {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": parameters
                    }
                    
                    agent_info["tools"].append(tool_info)
            
            registry_info[name] = agent_info
            
        return registry_info


def create_orchestrator_agent(registry: AgentRegistry) -> Agent[OrchestratorContext]:
    """
    Create an orchestrator agent that can route to specialized agents.
    
    Args:
        registry: The agent registry containing available agents
        
    Returns:
        An orchestrator agent configured for routing
    """
    # Create handoffs to all registered agents using the OpenAI Agents SDK
    handoffs_list = [handoff(entry.agent) for entry in registry.get_all_agents()]
    
    # Generate routing descriptions
    routing_text = "\n".join(registry.get_routing_descriptions())
    
    # Create dynamic instructions function that includes conversation history
    def dynamic_instructions(ctx: RunContextWrapper[OrchestratorContext], agent: Agent) -> str:
        # Get formatted conversation history
        history = ctx.context.memory.get_formatted_history(10)
        
        # Get permissions
        try:
            permissions = ", ".join(ctx.context.get_permissions())
        except Exception:
            permissions = "read, write, execute"
        
        # Build instructions with conversation history
        instructions = f"""You are an Orchestrator Agent that helps users by routing their requests to specialized agents.

You have access to these specialized agents:
{routing_text}

Your job is to:
1. Analyze each user query
2. Determine which specialized agent is best suited to handle it
3. Route the query to that agent using the appropriate handoff

When deciding which agent to use, consider the descriptions above.
For complex tasks that might need multiple steps or tools, consider using the planner agent to create a detailed execution plan.
If you're unsure which agent to use, ask the user for clarification.

When you hand off to another agent, briefly explain why you're doing so.

User permissions: {permissions}
Session ID: {ctx.context.session_id}

Recent conversation history:
{history}

Maintain continuity in the conversation by referencing previous interactions when appropriate.
"""
        return instructions
    
    # Create the orchestrator agent with dynamic instructions and handoffs
    instructions_with_handoff = prompt_with_handoff_instructions(dynamic_instructions)
    
    orchestrator = Agent[OrchestratorContext](
        name="Orchestrator",
        instructions=instructions_with_handoff,
        model="gpt-4o",
        handoffs=handoffs_list
    )
    
    return orchestrator


class AgentRunner(Generic[T]):
    """
    Runner for interactive sessions with orchestrated agents
    that uses the OpenAI Agents SDK under the hood
    """
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.context_map: Dict[str, Any] = {}
        self.orchestrator_context = OrchestratorContext()
        self.last_result = None  # Store the last result for conversation continuity
        
        # Create orchestrator
        self.orchestrator = create_orchestrator_agent(self.registry)
    
    def _get_context(self, agent_name: str) -> Optional[Any]:
        """Get or create context for an agent"""
        if agent_name not in self.context_map:
            entry = self.registry.get_agent(agent_name)
            if entry.context_factory:
                self.context_map[agent_name] = entry.context_factory()
        
        return self.context_map.get(agent_name)
    
    def _prepare_input_from_memory(self) -> List[Dict[str, Any]]:
        """
        Convert conversation memory to OpenAI message format for input to agents.
        
        Returns:
            List of message objects in OpenAI format
        """
        return self.orchestrator_context.memory.to_openai_messages()
        
    async def run_agent(self, agent_name: str, query: str) -> str:
        """
        Run a specific agent with a query.
        
        Args:
            agent_name: Name of the agent to run
            query: User query to process
            
        Returns:
            Agent response
        """
        # Record user message in conversation memory
        self.orchestrator_context.memory.add_user_message(query)
        
        # Get agent and context
        entry = self.registry.get_agent(agent_name)
        context = self._get_context(agent_name)
        
        # If this is the planning agent, update its context with registry info
        if agent_name == "planner" and hasattr(context, "update_registry_info"):
            registry_info = self.registry.get_registry_info()
            context.update_registry_info(registry_info)
        
        # Create the input for this turn
        if self.last_result:
            # Use the previous result to maintain conversation history
            # Add the new user query to the previous conversation
            input_data = self.last_result.to_input_list() + [
                {"role": "user", "content": query}
            ]
        else:
            # First turn, use conversation memory
            input_data = self._prepare_input_from_memory()
            if not input_data or input_data[-1]["role"] != "user" or input_data[-1]["content"] != query:
                # Make sure the latest user query is included
                input_data.append({"role": "user", "content": query})
        
        # Use the OpenAI Agents SDK's tracing with consistent group_id
        with trace(
            workflow_name=f"{agent_name} Workflow",
            group_id=self.orchestrator_context.session_id
        ):
            # Run the agent using the OpenAI Agents SDK Runner
            result = await OpenAIRunner.run(
                entry.agent,
                context=context,
                input=input_data
            )
        
        # Store this result for the next turn
        self.last_result = result
        
        # Record agent response in conversation memory
        response = result.final_output
        self.orchestrator_context.memory.add_assistant_message(response, agent_name)
        
        return response
    
    async def run_orchestrator(self, query: str) -> str:
        """
        Run the orchestrator agent with a query.
        
        Args:
            query: User query to process
            
        Returns:
            Orchestrator or specialized agent response
        """
        # Record user message in conversation memory
        self.orchestrator_context.memory.add_user_message(query)
        
        # Create the input for this turn
        if self.last_result:
            # Use the previous result to maintain conversation history
            # Add the new user query to the previous conversation
            input_data = self.last_result.to_input_list() + [
                {"role": "user", "content": query}
            ]
        else:
            # First turn, use conversation memory
            input_data = self._prepare_input_from_memory()
            if not input_data or input_data[-1]["role"] != "user" or input_data[-1]["content"] != query:
                # Make sure the latest user query is included
                input_data.append({"role": "user", "content": query})
        
        # Use the OpenAI Agents SDK's tracing with consistent group_id
        with trace(
            workflow_name="Orchestrator Workflow",
            group_id=self.orchestrator_context.session_id
        ):
            # Run the orchestrator agent
            result = await OpenAIRunner.run(
                self.orchestrator,
                context=self.orchestrator_context,
                input=input_data
            )
        
        # Store this result for the next turn
        self.last_result = result
        
        # Record response in conversation memory
        response = result.final_output
        self.orchestrator_context.memory.add_assistant_message(response, "orchestrator")
        
        return response
    
    async def interactive_session(self):
        """
        Run an interactive session with the agent orchestration system.
        """
        print("Agent Orchestration System")
        print(f"Available agents: {', '.join(self.registry.agents.keys())}, or let the orchestrator decide")
        print("Type '@agent_name query' to use a specific agent")
        print("Type '@planner query' to create an execution plan using available tools and agents")
        print("Type 'exit' to quit\n")
        
        while True:
            # Get user input
            user_input = input("\n> ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting session")
                break
            
            try:
                # Check for direct agent selection
                if user_input.startswith('@'):
                    parts = user_input.split(' ', 1)
                    if len(parts) < 2:
                        print("Please provide a query after the agent name")
                        continue
                    
                    agent_name = parts[0][1:]  # Remove the @ symbol
                    query = parts[1]
                    
                    if agent_name not in self.registry.agents:
                        print(f"Unknown agent: {agent_name}")
                        print(f"Available agents: {', '.join(self.registry.agents.keys())}")
                        continue
                    
                    response = await self.run_agent(agent_name, query)
                    print(f"\n{response}")
                else:
                    # Use orchestrator to route the query
                    response = await self.run_orchestrator(user_input)
                    print(f"\n{response}")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """
    Main entry point for the runner script.
    """
    print("Starting OpenAI Agents Orchestration System")
    
    # Import agents - must be done here to avoid circular imports
    from functionality.search.web_search_agent import web_search_agent
    from functionality.shell.shell_agent import advanced_shell_agent, ShellContext
    from functionality.planning.planning_agent import planning_agent
    from functionality.planning.planning_context import PlanningContext
    
    # Create agent registry
    registry = AgentRegistry()
    
    # Register agents
    registry.register(
        name="shell",
        agent=advanced_shell_agent,
        description="Executes bash commands and helps with system operations",
        context_factory=lambda: ShellContext()
    )
    
    registry.register(
        name="search",
        agent=web_search_agent,
        description="Finds information on the internet using web search",
        context_factory=None  # No special context needed
    )
    
    # Register planning agent
    registry.register(
        name="planner",
        agent=planning_agent,
        description="Creates execution plans using available tools and agents",
        context_factory=lambda: PlanningContext()
    )
    
    # Create agent runner with the registry
    agent_runner = AgentRunner(registry)
    
    # Run interactive session
    asyncio.run(agent_runner.interactive_session())


if __name__ == "__main__":
    main()