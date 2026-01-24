from app.agent.base import BaseAgent
from app.agent.react import ReActAgent
from app.agent.swe import SWEAgent
from app.agent.toolcall import ToolCallAgent

try:
    from app.agent.browser import BrowserAgent
except Exception:
    BrowserAgent = None

try:
    from app.agent.mcp import MCPAgent
except Exception:
    MCPAgent = None


__all__ = [
    "BaseAgent",
    "ReActAgent",
    "SWEAgent",
    "ToolCallAgent",
]

if BrowserAgent is not None:
    __all__.append("BrowserAgent")
if MCPAgent is not None:
    __all__.append("MCPAgent")
