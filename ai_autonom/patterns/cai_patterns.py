"""
CAI Agentic Patterns Adapter
Ported from CAI framework to support Swarm, Hierarchical, and Parallel execution patterns.
"""

from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

class PatternType(Enum):
    """Enumeration of available pattern types."""
    PARALLEL = "parallel"
    SWARM = "swarm"
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    
    @classmethod
    def from_string(cls, value: str) -> 'PatternType':
        """Convert string to PatternType."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid pattern type: {value}. Valid types: {[t.value for t in cls]}")

@dataclass
class ParallelConfig:
    """Configuration for parallel execution"""
    agent_name: str
    unified_context: bool = True
    
    def __init__(self, agent_name: str, unified_context: bool = True):
        self.agent_name = agent_name
        self.unified_context = unified_context

@dataclass
class Pattern:
    """
    Unified pattern class that adapts behavior based on type.
    """
    name: str
    type: Union[PatternType, str]
    description: str = ""
    
    # Type-specific attributes
    configs: List[ParallelConfig] = field(default_factory=list)  # For parallel
    entry_agent: Optional[Any] = None  # For swarm
    agents: List[Any] = field(default_factory=list)  # For swarm/hierarchical
    root_agent: Optional[Any] = None  # For hierarchical
    sequence: List[Any] = field(default_factory=list)  # For sequential
    conditions: Dict[str, Any] = field(default_factory=dict)  # For conditional
    
    # Common configuration options
    max_concurrent: Optional[int] = None
    unified_context: bool = True
    timeout: Optional[float] = None
    retry_on_failure: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize pattern type and validate."""
        if isinstance(self.type, str):
            self.type = PatternType.from_string(self.type)
    
    # Type-specific methods
    def add_parallel_agent(self, agent: Union[str, ParallelConfig]) -> 'Pattern':
        """Add an agent for parallel execution."""
        if self.type != PatternType.PARALLEL:
            raise ValueError(f"add_parallel_agent only works for PARALLEL patterns")
        
        if isinstance(agent, str):
            agent = ParallelConfig(agent, unified_context=self.unified_context)
        
        self.configs.append(agent)
        return self
    
    def set_entry_agent(self, agent: Any) -> 'Pattern':
        """Set the entry agent for swarm patterns."""
        if self.type != PatternType.SWARM:
            raise ValueError(f"set_entry_agent only works for SWARM patterns")
        
        self.entry_agent = agent
        if agent not in self.agents:
            self.agents.append(agent)
        return self
    
    def set_root_agent(self, agent: Any) -> 'Pattern':
        """Set the root agent for hierarchical patterns."""
        if self.type != PatternType.HIERARCHICAL:
            raise ValueError(f"set_root_agent only works for HIERARCHICAL patterns")
        
        self.root_agent = agent
        if agent not in self.agents:
            self.agents.append(agent)
        return self
    
    def add_sequence_step(self, agent: Any, wait_for_previous: bool = True) -> 'Pattern':
        """Add a step to sequential execution."""
        if self.type != PatternType.SEQUENTIAL:
            raise ValueError(f"add_sequence_step only works for SEQUENTIAL patterns")
        
        self.sequence.append({
            "agent": agent,
            "wait_for_previous": wait_for_previous
        })
        return self
    
    def add_condition(self, condition_name: str, agent: Any, predicate: Optional[Callable] = None) -> 'Pattern':
        """Add a conditional branch."""
        if self.type != PatternType.CONDITIONAL:
            raise ValueError(f"add_condition only works for CONDITIONAL patterns")
        
        self.conditions[condition_name] = {
            "agent": agent,
            "predicate": predicate
        }
        return self
    
    # Generic add
    def add(self, item: Any) -> 'Pattern':
        """Generic add method that works based on pattern type."""
        if self.type == PatternType.PARALLEL:
            return self.add_parallel_agent(item)
        elif self.type == PatternType.SWARM:
            self.agents.append(item)
            return self
        elif self.type == PatternType.HIERARCHICAL:
            self.agents.append(item)
            return self
        elif self.type == PatternType.SEQUENTIAL:
            return self.add_sequence_step(item)
        
        return self

# Factory functions
def parallel_pattern(name: str, description: str = "", agents: Optional[List[str]] = None, **kwargs) -> Pattern:
    """Create a parallel execution pattern."""
    pattern = Pattern(name=name, type=PatternType.PARALLEL, description=description, **kwargs)
    if agents:
        for agent in agents:
            pattern.add_parallel_agent(agent)
    return pattern

def swarm_pattern(name: str, entry_agent: Any, description: str = "", agents: Optional[List[Any]] = None, **kwargs) -> Pattern:
    """Create a swarm collaboration pattern."""
    pattern = Pattern(name=name, type=PatternType.SWARM, description=description, **kwargs)
    pattern.set_entry_agent(entry_agent)
    if agents:
        pattern.agents.extend(agents)
    return pattern

def hierarchical_pattern(name: str, root_agent: Any, description: str = "", children: Optional[List[Any]] = None, **kwargs) -> Pattern:
    """Create a hierarchical pattern."""
    pattern = Pattern(name=name, type=PatternType.HIERARCHICAL, description=description, **kwargs)
    pattern.set_root_agent(root_agent)
    if children:
        pattern.agents.extend(children)
    return pattern