from dataclasses import dataclass
import subprocess
import uuid
import os
from typing import List, Optional

from agents import Agent, RunContextWrapper
from base_context import BaseAgentContext

from functionality.shell.bash_tool import bash_tool

# Define a context class that could be used across the application
@dataclass
class ShellContext(BaseAgentContext):
    """Shell context for executing commands safely"""
    
    def __post_init__(self):
        """Initialize shell-specific attributes"""
        super().__post_init__()
        
        # Set working directory
        if "working_directory" not in self.attributes:
            try:
                self.attributes["working_directory"] = subprocess.run(
                    ["pwd"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                ).stdout.strip() or os.getcwd()
            except Exception:
                self.attributes["working_directory"] = os.getcwd()
            
        # Set admin status
        if "is_admin" not in self.attributes:
            try:
                # Check if user is in sudo group
                groups_process = subprocess.run(
                    ["groups", self.user_id], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                groups = groups_process.stdout.strip().split(": ")[1].split(" ") if ":" in groups_process.stdout else []
                self.attributes["is_admin"] = any(g in ["sudo", "wheel", "admin"] for g in groups)
            except Exception:
                self.attributes["is_admin"] = False
    
    @property
    def working_directory(self) -> str:
        """Property accessor for working_directory attribute"""
        return self.get_attribute("working_directory", os.getcwd())
        
    @working_directory.setter
    def working_directory(self, value: str) -> None:
        """Property setter for working_directory attribute"""
        self.set_attribute("working_directory", value)
    
    @property
    def user_id(self) -> str:
        """Property accessor for user_id attribute"""
        return self.get_attribute("user_id", "user")
        
    @user_id.setter
    def user_id(self, value: str) -> None:
        """Property setter for user_id attribute"""
        self.set_attribute("user_id", value)
        
    @property
    def is_admin(self) -> bool:
        """Property accessor for is_admin attribute"""
        return self.get_attribute("is_admin", False)
        
    @is_admin.setter
    def is_admin(self, value: bool) -> None:
        """Property setter for is_admin attribute"""
        self.set_attribute("is_admin", value)
    
    def get_permissions(self) -> List[str]:
        """
        Get permissions for the current user by querying the system.
        Extends the base permissions with shell-specific permissions.
        """
        # Get base permissions from parent class
        base_permissions = super().get_permissions()
        shell_permissions = []
        
        try:
            # Add execute permission (most shell users have this)
            shell_permissions.append("execute")
            
            # Add sudo if user is admin
            if self.is_admin:
                shell_permissions.append("sudo")
                
        except Exception:
            # Fallback to hardcoded permissions if anything fails
            shell_permissions.append("execute")
                
        # Combine base and shell-specific permissions
        return list(set(base_permissions + shell_permissions))

# Example of a dynamic instructions function
def dynamic_shell_instructions(context: RunContextWrapper[ShellContext], agent: Agent[ShellContext]) -> str:
    permissions = ", ".join(context.context.get_permissions())
    return f"""You are a helpful Shell Assistant that can execute bash commands for user {context.context.user_id}.
    
Current working directory: {context.context.working_directory}
User permissions: {permissions}
Session ID: {context.context.session_id}

Use the available bash tool to execute commands and provide the results to the user.
    
Always explain what the command does before executing it.
Be security-conscious and don't execute potentially harmful commands.
If a command fails, explain the error and suggest corrections.

When providing file listings or command outputs, format them clearly for readability.
"""

# Advanced agent with dynamic instructions
advanced_shell_agent = Agent[ShellContext](
    name="Advanced Shell Assistant",
    instructions=dynamic_shell_instructions,
    model="gpt-4o",
    tools=[bash_tool],
)
