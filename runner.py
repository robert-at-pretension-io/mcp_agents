from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
import asyncio
from dataclasses import dataclass, field
import uuid
import os
import logging

# Configure logging to reduce httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Optional: Disable OpenAI Agents SDK tracing
# from agents.tracing import set_tracing_disabled
# set_tracing_disabled(True)

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
    Represents an agent entry in the registry with routing criteria and tags
    """
    name: str
    agent: Agent
    description: str
    tags: List[str] = field(default_factory=list)
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
        tags: List[str] = None,
        context_factory: Optional[Callable[[], Any]] = None
    ) -> None:
        """
        Register an agent with the registry.
        
        Args:
            name: Unique identifier for the agent
            agent: The agent instance
            description: Description of when to use this agent
            tags: List of capability tags for this agent (e.g., ["search", "data", "execution"])
            context_factory: Optional function to create a context object for this agent
        """
        if name in self.agents:
            raise ValueError(f"Agent with name '{name}' already registered")
        
        # Use empty list if tags is None
        tags = tags or []
        
        self.agents[name] = AgentRegistryEntry(
            name=name,
            agent=agent,
            description=description,
            tags=tags,
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
        
    def find_agents_by_tags(self, tags: List[str], match_all: bool = False) -> List[AgentRegistryEntry]:
        """
        Find agents that match the specified tags.
        
        Args:
            tags: List of tags to match
            match_all: If True, agents must have all specified tags; if False, agents must have at least one
            
        Returns:
            List of agent entries that match the tag criteria
        """
        if not tags:
            return []
            
        results = []
        
        for entry in self.agents.values():
            # Check if entry has any of the requested tags
            if match_all:
                # All specified tags must be present
                if all(tag in entry.tags for tag in tags):
                    results.append(entry)
            else:
                # At least one tag must be present
                if any(tag in entry.tags for tag in tags):
                    results.append(entry)
                    
        return results
    
    def get_routing_descriptions(self) -> List[str]:
        """
        Get a list of agent descriptions for routing.
        
        Returns:
            List of agent descriptions
        """
        return [f"- {entry.name}: {entry.description}" for entry in self.agents.values()]
        
    def get_tag_summary(self) -> str:
        """
        Generate a summary of capabilities organized by tags.
        
        Returns:
            Formatted string of capabilities grouped by tags
        """
        # Create a mapping of tags to agents
        tag_to_agents = {}
        
        for entry in self.agents.values():
            for tag in entry.tags:
                if tag not in tag_to_agents:
                    tag_to_agents[tag] = []
                tag_to_agents[tag].append({
                    "name": entry.name,
                    "description": entry.description
                })
        
        # Generate the summary
        if not tag_to_agents:
            return "No tagged capabilities available."
            
        summary_parts = []
        
        # Sort tags alphabetically for consistent output
        for tag in sorted(tag_to_agents.keys()):
            agents = tag_to_agents[tag]
            summary_parts.append(f"## {tag.upper()} CAPABILITIES:")
            
            for agent_info in agents:
                summary_parts.append(f"- {agent_info['name']}: {agent_info['description']}")
                
            summary_parts.append("")  # Empty line between tags
        
        return "\n".join(summary_parts)
        
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
                "tags": entry.tags,
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
    # Create handoffs to all registered agents using the OpenAI Agents SDK -- this should NOT be used when the orchestrator is "in charge". Otherwise, other agents can pre-maturely end the conversation
    # handoffs_list = [handoff(entry.agent) for entry in registry.get_all_agents()]
    
    # Create dynamic instructions function that includes conversation history and tag-based capabilities
    def dynamic_instructions(ctx: RunContextWrapper[OrchestratorContext], agent: Agent) -> str:
        # Get formatted conversation history
        history = ctx.context.memory.get_formatted_history(10)
        
        # Get tag-based capability summary
        tag_summary = registry.get_tag_summary()
        
        # Get permissions
        try:
            permissions = ", ".join(ctx.context.get_permissions())
        except Exception:
            permissions = "read, write, execute"
        
        # Build instructions with conversation history and tag-based capabilities
        instructions = f"""You are an Orchestrator Agent that maximizes value through strategic multi-tool utilization.

You have access to a wide range of specialized tools and agents with these capabilities:
{tag_summary}

Your primary directive:
1. Analyze each user query thoroughly to identify ALL required capabilities
2. For complex requests, ALWAYS use MULTIPLE relevant tools in a SINGLE response
3. Chain tools together - use the output of one tool as input to another
4. Prioritize efficiency by using the minimal sufficient set of tools to solve the problem completely

Remember that complex problems are rarely solved with a single tool. Look for opportunities to:
- Use the planner to create structured execution plans for multi-step tasks
- Combine data retrieval with data processing
- Pair knowledge discovery with execution capabilities 
- Follow planning with immediate execution steps

User permissions: {permissions}
Session ID: {ctx.context.session_id}

Recent conversation history:
{history}

Maintain continuity in the conversation by referencing previous interactions when appropriate.
"""
        return instructions
    
    # Create the orchestrator agent with dynamic instructions and handoffs
    instructions_with_handoff = prompt_with_handoff_instructions(dynamic_instructions)
    
    model_settings = ModelSettings(tool_choice='use_planner')

    orchestrator = Agent[OrchestratorContext](
        name="Orchestrator",
        instructions=instructions_with_handoff,
        model="gpt-4o",
        # handoffs=handoffs_list, # Don't use because we already gave the orchestrator all the agents as tools.
        model_settings=model_settings
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
        # Note: All agents will be registered as tools for the orchestrator in main()
    
    def _get_context(self, agent_name: str) -> Optional[Any]:
        """Get or create context for an agent"""
        if agent_name not in self.context_map:
            entry = self.registry.get_agent(agent_name)
            if entry.context_factory:
                # Create a new context using the factory function
                self.context_map[agent_name] = entry.context_factory()
            else:
                # No context factory, return None
                return None
        
        # Return the cached context
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
        if agent_name == "planner":
            if context is not None and hasattr(context, "update_registry_info"):
                try:
                    registry_info = self.registry.get_registry_info()
                    # Log information about the registry info
                    agent_count = len(registry_info)
                    tool_counts = {name: len(info.get('tools', [])) for name, info in registry_info.items()}
                    logging.info(f"Registry info has {agent_count} agents with tool counts: {tool_counts}")
                        
                    # Check if there are no tools available
                    if all(len(info.get('tools', [])) == 0 for info in registry_info.values()):
                        logging.warning("No tools found in the registry for the planning agent to use")
                        
                    # Call the update_registry_info method on the context object
                    context.update_registry_info(registry_info)
                    logging.info(f"Updated planning context registry info with {len(registry_info)} agents")
                except Exception as e:
                    logging.error(f"Failed to update registry info: {e}")
            else:
                logging.warning("Planning agent context is None or doesn't have update_registry_info method")
        
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
        
        For complex queries, this will first use the planner agent to create a plan
        before proceeding with the orchestrator to execute the plan.
        
        Args:
            query: User query to process
            
        Returns:
            Orchestrator response
        """
        # Record user message in conversation memory
        self.orchestrator_context.memory.add_user_message(query)
    
        # For simple queries, use regular orchestration without planning
        if self.last_result:
            input_data = self.last_result.to_input_list() + [
                {"role": "user", "content": query}
            ]
        else:
            input_data = self._prepare_input_from_memory()
            if not input_data or input_data[-1]["role"] != "user" or input_data[-1]["content"] != query:
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
        print("Agent Orchestration System with Tag-Based Capabilities")
        print(f"Available agents: {', '.join(self.registry.agents.keys())}, or let the orchestrator decide")
        print("\nCapability Tags:")
        
        # Create a set of all unique tags
        all_tags = set()
        for entry in self.registry.agents.values():
            all_tags.update(entry.tags)
            
        # Display tags grouped by agent
        for name, entry in self.registry.agents.items():
            if entry.tags:
                print(f"  - {name}: [{', '.join(entry.tags)}]")
        
        print("\nUsage:")
        print("  Type '@agent_name query' to use a specific agent")
        print("  Type '@planner query' to create a multi-step execution plan")
        print("  Or simply type your query to let the orchestrator select the best tools")
        print("  Type 'exit' to quit\n")
        
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
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    print("Starting OpenAI Agents Orchestration System")
    
    # Import agents - must be done here to avoid circular imports
    from functionality.search.web_search_agent import web_search_agent
    from functionality.shell.shell_agent import advanced_shell_agent, ShellContext
    from functionality.planning.planning_agent import planning_agent
    from functionality.planning.planning_context import PlanningContext
    from functionality.scraping.scraping_agent import scraping_agent
    
    # Create agent registry
    registry = AgentRegistry()
    
    # Register agents with tags
    registry.register(
        name="shell",
        agent=advanced_shell_agent,
        description="Executes bash commands and helps with system operations",
        tags=["execution", "system", "files", "commands"],
        context_factory=lambda: ShellContext()
    )
    
    registry.register(
        name="search",
        agent=web_search_agent,
        description="Finds information on the internet using web search",
        tags=["search", "web", "knowledge", "information"],
        context_factory=None  # No special context needed
    )
    
    # Register planning agent
    registry.register(
        name="planner",
        agent=planning_agent,
        description="Creates execution plans using available tools and agents",
        tags=["planning", "orchestration", "strategy", "coordination"],
        context_factory=lambda: PlanningContext()
    )
    
    # Register scraping agent
    registry.register(
        name="scraper",
        agent=scraping_agent,
        description="Extracts and processes content from websites",
        tags=["web", "information", "extraction", "research", "content", "scraping"],
        context_factory=None  # No special context needed
    )
    
    # Function to register all agents as tools for the orchestrator
    def register_all_as_tools(orchestrator: Agent, registry: AgentRegistry):
        """Register all agents as tools for the orchestrator"""
        for name, entry in registry.agents.items():
            # Skip the orchestrator itself
            if name == "orchestrator":
                continue
                
            # Create tool description that includes tags
            tag_desc = f"[Tags: {', '.join(entry.tags)}]"
            tool_description = f"{entry.description} {tag_desc}"
            
            # Register the agent as a tool
            tool = entry.agent.as_tool(
                tool_name=f"use_{name}",  # This is the name that must match in tool_choice
                tool_description=tool_description
            )
            
            # Add to orchestrator's tools
            orchestrator.tools.append(tool)
    
    # Create agent runner with the registry
    agent_runner = AgentRunner(registry)
    
    # Register all agents as tools for the orchestrator
    register_all_as_tools(agent_runner.orchestrator, registry)
    
    # Run interactive session
    asyncio.run(agent_runner.interactive_session())


if __name__ == "__main__":
    main()
