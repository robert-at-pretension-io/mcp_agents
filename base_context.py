from typing import List, Dict, Any, Optional
import uuid
from dataclasses import dataclass, field
import subprocess
import os
from agents import RunContextWrapper

@dataclass
class BaseAgentContext:
    """
    Base class for all agent contexts to ensure consistent interface.
    
    This context serves as a dependency injection mechanism,
    similar to how the OpenAI Agents SDK uses context objects.
    """
    session_id: str = field(default_factory=lambda: f"thread_{uuid.uuid4().hex[:8]}")
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize common attributes that many agent contexts might need"""
        # Set user ID if not already set
        if "user_id" not in self.attributes:
            try:
                self.attributes["user_id"] = subprocess.run(
                    ["whoami"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                ).stdout.strip() or "user"
            except Exception:
                self.attributes["user_id"] = "user"
    
    def get_permissions(self) -> List[str]:
        """
        Get default permissions for the context.
        Can be overridden by subclasses for specific permission sets.
        """
        return ["read", "write"]
        
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute by key with an optional default value"""
        return self.attributes.get(key, default)
        
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute by key and value"""
        self.attributes[key] = value
        
    @property
    def user_id(self) -> str:
        """Property accessor for user_id attribute"""
        return self.get_attribute("user_id", "user")
    
    @property
    def working_directory(self) -> str:
        """Property accessor for working_directory attribute, with fallback"""
        return self.get_attribute("working_directory", os.getcwd())

    @staticmethod
    def from_run_context(ctx: RunContextWrapper) -> 'BaseAgentContext':
        """
        Create a BaseAgentContext from a RunContextWrapper.
        This is useful for interoperability with the OpenAI Agents SDK.
        """
        if hasattr(ctx, 'context') and isinstance(ctx.context, BaseAgentContext):
            return ctx.context
            
        # Create a new context if the incoming one isn't a BaseAgentContext
        context = BaseAgentContext()
        
        # Try to copy any available attributes
        if hasattr(ctx, 'context') and hasattr(ctx.context, '__dict__'):
            for key, value in ctx.context.__dict__.items():
                context.set_attribute(key, value)
                
        return context