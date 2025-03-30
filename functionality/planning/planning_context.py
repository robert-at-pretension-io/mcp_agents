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
        
        # Store agent registry information
        self.agent_registry = {}
    
    def update_registry_info(self, registry_info: Dict[str, Any]) -> None:
        """
        Update agent registry information in the context
        
        Args:
            registry_info: Dictionary mapping agent names to their metadata
        """
        self.agent_registry = registry_info