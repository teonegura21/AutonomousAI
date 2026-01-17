#!/usr/bin/env python3
"""
Agent Message Bus
Inter-agent communication system - agents can send messages, ask questions, broadcast
"""

import threading
import queue
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Types of messages between agents"""
    MESSAGE = "message"
    QUESTION = "question"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class AgentMessage:
    """A message between agents"""
    from_agent: str
    to_agent: str  # "*" for broadcast
    content: Dict[str, Any]
    message_type: MessageType
    topic: Optional[str] = None
    correlation_id: Optional[str] = None  # For request-response matching
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "message_type": self.message_type.value,
            "topic": self.topic,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }


class AgentMessageBus:
    """
    Central message bus for inter-agent communication.
    Agents communicate via messages, not just sequential execution.
    This enables collaboration: "What coordinate system did you choose?"
    """
    
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}  # agent_id -> message queue
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> [agent_ids]
        self.message_history: List[AgentMessage] = []
        self.callbacks: Dict[str, Callable] = {}  # agent_id -> callback function
        self.lock = threading.Lock()
        self._message_counter = 0
    
    def register_agent(self, agent_id: str, callback: Optional[Callable] = None) -> None:
        """
        Register agent to receive messages
        
        Args:
            agent_id: Unique agent identifier
            callback: Optional callback function for async message handling
        """
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = queue.Queue()
                print(f"[MSG_BUS] Agent registered: {agent_id}")
            
            if callback:
                self.callbacks[agent_id] = callback
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        with self.lock:
            self.queues.pop(agent_id, None)
            self.callbacks.pop(agent_id, None)
            # Remove from all subscriptions
            for topic in self.subscriptions:
                if agent_id in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(agent_id)
            print(f"[MSG_BUS] Agent unregistered: {agent_id}")
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID"""
        with self.lock:
            self._message_counter += 1
            return f"msg_{self._message_counter}_{datetime.now().strftime('%H%M%S')}"
    
    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: Dict[str, Any],
        msg_type: MessageType = MessageType.MESSAGE
    ) -> bool:
        """
        Send direct message to another agent
        
        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            content: Message content
            msg_type: Type of message
        
        Returns:
            True if message was delivered
        """
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type=msg_type,
            correlation_id=self._generate_correlation_id()
        )
        
        with self.lock:
            self.message_history.append(msg)
            
            if to_agent in self.queues:
                self.queues[to_agent].put(msg)
                print(f"[MSG_BUS] {from_agent} -> {to_agent}: {msg_type.value}")
                
                # Trigger callback if registered
                if to_agent in self.callbacks:
                    try:
                        self.callbacks[to_agent](msg)
                    except Exception as e:
                        print(f"[MSG_BUS] Callback error: {e}")
                
                return True
            else:
                print(f"[MSG_BUS] Agent not found: {to_agent}")
                return False
    
    def broadcast(
        self,
        from_agent: str,
        topic: str,
        content: Dict[str, Any]
    ) -> int:
        """
        Broadcast to all subscribers of a topic
        
        Args:
            from_agent: Sender agent ID
            topic: Topic to broadcast to
            content: Message content
        
        Returns:
            Number of recipients
        """
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent="*",
            content=content,
            message_type=MessageType.BROADCAST,
            topic=topic,
            correlation_id=self._generate_correlation_id()
        )
        
        with self.lock:
            self.message_history.append(msg)
            
            subscribers = self.subscriptions.get(topic, [])
            delivered = 0
            
            for agent_id in subscribers:
                if agent_id != from_agent and agent_id in self.queues:
                    self.queues[agent_id].put(msg)
                    delivered += 1
            
            print(f"[MSG_BUS] Broadcast from {from_agent} on '{topic}': {delivered} recipients")
            return delivered
    
    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to a topic"""
        with self.lock:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = []
            if agent_id not in self.subscriptions[topic]:
                self.subscriptions[topic].append(agent_id)
                print(f"[MSG_BUS] {agent_id} subscribed to '{topic}'")
    
    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from a topic"""
        with self.lock:
            if topic in self.subscriptions and agent_id in self.subscriptions[topic]:
                self.subscriptions[topic].remove(agent_id)
    
    def query_agent(
        self,
        asking_agent: str,
        target_agent: str,
        question: str,
        timeout: float = 30.0
    ) -> Optional[str]:
        """
        Ask another agent a question and wait for response
        
        Args:
            asking_agent: Agent asking the question
            target_agent: Agent to ask
            question: The question to ask
            timeout: How long to wait for response
        
        Returns:
            Response content or None if timeout
        """
        correlation_id = self._generate_correlation_id()
        
        # Send question
        msg = AgentMessage(
            from_agent=asking_agent,
            to_agent=target_agent,
            content={"question": question},
            message_type=MessageType.QUESTION,
            correlation_id=correlation_id
        )
        
        with self.lock:
            self.message_history.append(msg)
            if target_agent in self.queues:
                self.queues[target_agent].put(msg)
        
        print(f"[MSG_BUS] Question from {asking_agent} to {target_agent}")
        
        # Wait for response
        try:
            start_time = datetime.now()
            while True:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
                
                if asking_agent in self.queues:
                    try:
                        response = self.queues[asking_agent].get(timeout=1.0)
                        if (response.message_type == MessageType.RESPONSE and
                            response.correlation_id == correlation_id):
                            return response.content.get("answer", "")
                        else:
                            # Put it back if not the response we're waiting for
                            self.queues[asking_agent].put(response)
                    except queue.Empty:
                        continue
                else:
                    return None
                    
        except Exception as e:
            print(f"[MSG_BUS] Query error: {e}")
            return None
    
    def respond_to_question(
        self,
        responding_agent: str,
        original_message: AgentMessage,
        answer: str
    ) -> bool:
        """
        Respond to a question from another agent
        
        Args:
            responding_agent: Agent sending the response
            original_message: The original question message
            answer: The answer to the question
        
        Returns:
            True if response was sent
        """
        msg = AgentMessage(
            from_agent=responding_agent,
            to_agent=original_message.from_agent,
            content={"answer": answer},
            message_type=MessageType.RESPONSE,
            correlation_id=original_message.correlation_id
        )
        
        with self.lock:
            self.message_history.append(msg)
            if original_message.from_agent in self.queues:
                self.queues[original_message.from_agent].put(msg)
                return True
        return False
    
    def get_messages(
        self,
        agent_id: str,
        block: bool = False,
        timeout: float = 1.0
    ) -> List[AgentMessage]:
        """
        Get all pending messages for an agent
        
        Args:
            agent_id: Agent to get messages for
            block: Whether to block waiting for messages
            timeout: How long to block
        
        Returns:
            List of messages
        """
        messages = []
        
        if agent_id not in self.queues:
            return messages
        
        q = self.queues[agent_id]
        
        if block and q.empty():
            try:
                msg = q.get(timeout=timeout)
                messages.append(msg)
            except queue.Empty:
                pass
        
        # Get all remaining messages
        while not q.empty():
            try:
                messages.append(q.get_nowait())
            except queue.Empty:
                break
        
        return messages
    
    def peek_messages(self, agent_id: str) -> int:
        """Get count of pending messages without removing them"""
        if agent_id in self.queues:
            return self.queues[agent_id].qsize()
        return 0
    
    def get_history(
        self,
        agent_id: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get message history, optionally filtered"""
        with self.lock:
            history = self.message_history.copy()
        
        if agent_id:
            history = [
                m for m in history
                if m.from_agent == agent_id or m.to_agent == agent_id
            ]
        
        if topic:
            history = [m for m in history if m.topic == topic]
        
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        with self.lock:
            return {
                "registered_agents": list(self.queues.keys()),
                "topics": list(self.subscriptions.keys()),
                "total_messages": len(self.message_history),
                "pending_messages": {
                    agent: q.qsize() for agent, q in self.queues.items()
                },
                "subscriptions": {
                    topic: len(agents) for topic, agents in self.subscriptions.items()
                }
            }
    
    def clear_history(self) -> None:
        """Clear message history"""
        with self.lock:
            self.message_history.clear()


# Global message bus instance
_global_bus: Optional[AgentMessageBus] = None


def get_global_bus() -> AgentMessageBus:
    """Get the global message bus"""
    global _global_bus
    if _global_bus is None:
        _global_bus = AgentMessageBus()
    return _global_bus


if __name__ == "__main__":
    # Test message bus
    bus = AgentMessageBus()
    
    print("\n" + "="*60)
    print("AGENT MESSAGE BUS TEST")
    print("="*60 + "\n")
    
    # Register agents
    bus.register_agent("orchestrator")
    bus.register_agent("coder_qwen")
    bus.register_agent("tester_phi")
    
    # Subscribe to topics
    bus.subscribe("coder_qwen", "code_updates")
    bus.subscribe("tester_phi", "code_updates")
    
    # Send direct message
    bus.send_message(
        "orchestrator",
        "coder_qwen",
        {"task": "Write a factorial function"},
        MessageType.MESSAGE
    )
    
    # Broadcast
    bus.broadcast(
        "coder_qwen",
        "code_updates",
        {"file": "factorial.py", "status": "completed"}
    )
    
    # Get messages
    messages = bus.get_messages("coder_qwen")
    print(f"\nMessages for coder_qwen: {len(messages)}")
    for m in messages:
        print(f"  - From {m.from_agent}: {m.content}")
    
    messages = bus.get_messages("tester_phi")
    print(f"\nMessages for tester_phi: {len(messages)}")
    for m in messages:
        print(f"  - From {m.from_agent}: {m.content}")
    
    # Stats
    print(f"\nBus stats: {bus.get_stats()}")
