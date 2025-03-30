from dataclasses import dataclass, field
from typing import Dict, Any

from base_context import BaseAgentContext

@dataclass
class PlanningContext(BaseAgentContext):
    """Context for the planning agent"""
    
    def __post_init__(self):
        """Initialize planning-specific attributes"""
        super().__post_init__()
        self.set_attribute("role", "planner")
        
        # Store agent registry information in attributes
        # This makes it accessible to the OpenAI Agents SDK
        self.set_attribute("agent_registry", {})
    
    @property
    def agent_registry(self) -> Dict[str, Any]:
        """Get the agent registry from attributes"""
        return self.get_attribute("agent_registry", {})
    
    @agent_registry.setter
    def agent_registry(self, value: Dict[str, Any]) -> None:
        """Set the agent registry in attributes"""
        self.set_attribute("agent_registry", value)
    
    def update_registry_info(self, registry_info: Dict[str, Any]) -> None:
        """
        Update agent registry information in the context
        
        Args:
            registry_info: Dictionary mapping agent names to their metadata
        """
        import logging
        logging.info(f"Updating planning context with registry info containing {len(registry_info)} agents")
        
        # Store in attributes to make it accessible to the SDK
        self.agent_registry = registry_info
        
        # Log the received registry info for debugging
        for agent_name, agent_info in registry_info.items():
            tools = agent_info.get('tools', [])
            logging.info(f"Agent {agent_name}: {len(tools)} tools registered")
            for tool in tools:
                tool_name = tool.get('name', 'unnamed')
                logging.info(f"  - Tool: {tool_name}")