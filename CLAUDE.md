- Multi-Tool Utilization with Tags

Here's the redesigned approach using tags instead of categories, allowing agents to have multiple capabilities and classifications:

## 1. Create a Tag-Based Dynamic Discovery System

```python
def dynamic_instructions(ctx: RunContextWrapper[OrchestratorContext], agent: Agent) -> str:
    # Get formatted conversation history
    history = ctx.context.memory.get_formatted_history(10)
    
    # Get tag-based capability summary
    tag_summary = registry.get_tag_summary()
    
    # Get permissions
    permissions = ", ".join(ctx.context.get_permissions())
    
    instructions = f"""You are an Orchestrator Agent that maximizes value through strategic multi-tool utilization.

You have access to a wide range of specialized tools and agents with these capabilities:
{tag_summary}

Your primary directive:
1. Analyze each user query thoroughly to identify ALL required capabilities
2. For complex requests, ALWAYS use MULTIPLE relevant tools in a SINGLE response
3. Chain tools together - use the output of one tool as input to another
4. Prioritize efficiency by using the minimal sufficient set of tools to solve the problem completely

IMPORTANT: Do not limit yourself to just one tool per response. Effective orchestration means combining complementary capabilities within the same turn.

Remember that complex problems are rarely solved with a single tool. Look for opportunities to:
- Combine data retrieval with data processing
- Pair knowledge discovery with execution capabilities
- Use planning capabilities followed by immediate execution steps

User permissions: {permissions}
Session ID: {ctx.context.session_id}

Recent conversation history:
{history}
"""
    return instructions
```

## 2. Implement a Tag-Based Registry System

```python
class AgentRegistryEntry:
    """
    Represents an agent entry in the registry with routing criteria and ta[Pasted text +40 lines] [Pasted text +234 lines]  # Get parameter schema if available
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
        
    planning_context.update_registry_info(registry_info)
```

## Summary

This tag-based approach offers several advantages over the category-based system:

1. **Multi-dimensional classification** - Agents can belong to multiple capability groups
2. **Flexible discovery** - Find agents by any combination of tags
3. **Richer metadata** - Tags provide more contextual information about agent capabilities
4. **Better matching** - Easier to match user queries to relevant agent capabilities
5. **Extensible** - New tags can be added without restructuring the entire system

By using tags instead of categories, your orchestrator can more effectively identify complementary capabilities and combine them in intelligent ways. This approach scales exceptionally well to hundreds or thousands of agents while maintaining the ability to find optimal tool combinations for complex tasks.