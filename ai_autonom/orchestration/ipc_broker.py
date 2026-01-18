#!/usr/bin/env python3
"""
Inter-Process Communication Broker

SQLite-based IPC for cross-thread/process communication with push notifications.
Provides pub/sub messaging, shared context storage, and request/response patterns.

Architecture:
- SQLite as persistent message queue and shared state store
- Push notifications via callbacks registered per channel
- Thread-safe with connection pooling
- Process-safe with file locking

Usage:
    from ai_autonom.orchestration.ipc_broker import IPCBroker, get_broker
    
    broker = get_broker()
    
    # Publish message
    broker.publish("scan_complete", {"target": "10.0.0.1", "results": [...]})
    
    # Subscribe with callback
    def on_scan(msg):
        print(f"Scan received: {msg}")
    broker.subscribe("scan_complete", on_scan)
    
    # Shared context
    broker.set_shared("current_target", "10.0.0.1")
    target = broker.get_shared("current_target")
"""

import sqlite3
import json
import time
import threading
import uuid
import os
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from contextlib import contextmanager


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class IPCMessage:
    """Message structure for IPC communication."""
    id: str
    channel: str
    sender_id: str
    payload: Dict[str, Any]
    priority: int = MessagePriority.NORMAL.value
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None
    processed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCMessage':
        return cls(**data)


class IPCBroker:
    """
    SQLite-based Inter-Process Communication Broker.
    
    Features:
    - Pub/Sub messaging with push callbacks
    - Persistent shared context store
    - Request/Response pattern (RPC-like)
    - Thread-safe with per-thread connections
    - Process-safe with file locking
    - Message expiration and cleanup
    
    Example:
        broker = IPCBroker()
        
        # Pub/Sub
        broker.subscribe("events", lambda msg: print(msg))
        broker.publish("events", {"type": "scan_complete"})
        
        # Shared context
        broker.set_shared("target_ip", "10.0.0.1")
        ip = broker.get_shared("target_ip")
        
        # Request/Response
        response = broker.request("agent_1", {"action": "scan"}, timeout=30)
    """
    
    def __init__(self, db_path: str = ".runtime/data/ipc_messages.db"):
        """
        Initialize IPC Broker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Subscription registry: channel -> list of callbacks
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._subscription_lock = threading.Lock()
        
        # Response waiters: correlation_id -> threading.Event
        self._response_waiters: Dict[str, threading.Event] = {}
        self._responses: Dict[str, IPCMessage] = {}
        self._response_lock = threading.Lock()
        
        # Background polling thread
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._last_message_id = 0
        
        # Process ID for sender identification
        self._process_id = f"pid_{os.getpid()}_tid_{threading.current_thread().ident}"
        
        # Initialize database
        self._init_db()
        
        # Start polling for push notifications
        self._start_polling()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
        return self._local.conn
    
    @contextmanager
    def _locked_write(self):
        """Context manager for locked write operations."""
        lock_path = str(self.db_path) + ".lock"
        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        
        conn.executescript("""
            -- Messages table for pub/sub
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL,
                channel TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                correlation_id TEXT,
                reply_to TEXT,
                timestamp TEXT NOT NULL,
                expires_at TEXT,
                processed INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel);
            CREATE INDEX IF NOT EXISTS idx_messages_correlation ON messages(correlation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_processed ON messages(processed);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            
            -- Shared context store
            CREATE TABLE IF NOT EXISTS shared_context (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                context_type TEXT DEFAULT 'json',
                owner_id TEXT,
                version INTEGER DEFAULT 1,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            );
            
            -- Channel subscriptions (for persistent subscriptions)
            CREATE TABLE IF NOT EXISTS subscriptions (
                subscriber_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (subscriber_id, channel)
            );
            
            -- Vector context index (for semantic search)
            CREATE TABLE IF NOT EXISTS vector_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_key TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_vector_key ON vector_context(context_key);
        """)
        conn.commit()
    
    # =========================================================================
    # PUB/SUB METHODS
    # =========================================================================
    
    def publish(
        self,
        channel: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        expires_seconds: Optional[int] = None
    ) -> str:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name to publish to
            payload: Message payload (dict)
            priority: Message priority
            correlation_id: For request/response correlation
            reply_to: Channel to send response to
            expires_seconds: Message expiration time
        
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        expires_at = None
        if expires_seconds:
            expires_at = datetime.fromtimestamp(
                time.time() + expires_seconds
            ).isoformat()
        
        with self._locked_write():
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO messages (
                    message_id, channel, sender_id, payload, priority,
                    correlation_id, reply_to, timestamp, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id, channel, self._process_id,
                json.dumps(payload), priority.value if isinstance(priority, MessagePriority) else priority,
                correlation_id, reply_to,
                datetime.now().isoformat(), expires_at
            ))
            conn.commit()
        
        # Trigger immediate notification to local subscribers
        self._notify_local_subscribers(channel, IPCMessage(
            id=message_id,
            channel=channel,
            sender_id=self._process_id,
            payload=payload,
            priority=priority.value if isinstance(priority, MessagePriority) else priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            expires_at=expires_at
        ))
        
        return message_id
    
    def subscribe(
        self,
        channel: str,
        callback: Callable[[IPCMessage], None],
        persistent: bool = False
    ) -> None:
        """
        Subscribe to a channel with a callback.
        
        Args:
            channel: Channel to subscribe to
            callback: Function to call when message arrives
            persistent: If True, store subscription in database
        """
        with self._subscription_lock:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = []
            self._subscriptions[channel].append(callback)
        
        if persistent:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions (subscriber_id, channel, created_at)
                VALUES (?, ?, ?)
            """, (self._process_id, channel, datetime.now().isoformat()))
            conn.commit()
    
    def unsubscribe(self, channel: str, callback: Optional[Callable] = None) -> None:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel to unsubscribe from
            callback: Specific callback to remove (None = all)
        """
        with self._subscription_lock:
            if channel in self._subscriptions:
                if callback:
                    self._subscriptions[channel] = [
                        c for c in self._subscriptions[channel] if c != callback
                    ]
                else:
                    del self._subscriptions[channel]
    
    def _notify_local_subscribers(self, channel: str, message: IPCMessage) -> None:
        """Notify all local subscribers of a message."""
        with self._subscription_lock:
            callbacks = self._subscriptions.get(channel, [])
        
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"[IPC] Callback error on {channel}: {e}")
    
    def get_messages(
        self,
        channel: str,
        since_id: Optional[int] = None,
        limit: int = 100,
        mark_processed: bool = True
    ) -> List[IPCMessage]:
        """
        Get messages from a channel.
        
        Args:
            channel: Channel to read from
            since_id: Only messages after this ID
            limit: Maximum messages to return
            mark_processed: Mark messages as processed
        
        Returns:
            List of messages
        """
        conn = self._get_conn()
        
        query = """
            SELECT id, message_id, channel, sender_id, payload, priority,
                   correlation_id, reply_to, timestamp, expires_at, processed
            FROM messages
            WHERE channel = ? AND processed = 0
        """
        params = [channel]
        
        if since_id:
            query += " AND id > ?"
            params.append(since_id)
        
        # Filter expired messages
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(datetime.now().isoformat())
        
        query += " ORDER BY priority DESC, id ASC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        
        messages = []
        for row in rows:
            msg = IPCMessage(
                id=row['message_id'],
                channel=row['channel'],
                sender_id=row['sender_id'],
                payload=json.loads(row['payload']),
                priority=row['priority'],
                correlation_id=row['correlation_id'],
                reply_to=row['reply_to'],
                timestamp=row['timestamp'],
                expires_at=row['expires_at'],
                processed=bool(row['processed'])
            )
            messages.append(msg)
            
            if mark_processed:
                conn.execute(
                    "UPDATE messages SET processed = 1 WHERE id = ?",
                    (row['id'],)
                )
        
        if mark_processed and messages:
            conn.commit()
        
        return messages
    
    # =========================================================================
    # REQUEST/RESPONSE METHODS
    # =========================================================================
    
    def request(
        self,
        target_channel: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[IPCMessage]:
        """
        Send request and wait for response (RPC pattern).
        
        Args:
            target_channel: Channel to send request to
            payload: Request payload
            timeout: Seconds to wait for response
        
        Returns:
            Response message or None if timeout
        """
        correlation_id = str(uuid.uuid4())
        reply_channel = f"_reply_{self._process_id}_{correlation_id}"
        
        # Set up response waiter
        event = threading.Event()
        with self._response_lock:
            self._response_waiters[correlation_id] = event
        
        # Subscribe to reply channel
        def handle_response(msg: IPCMessage):
            with self._response_lock:
                self._responses[correlation_id] = msg
                if correlation_id in self._response_waiters:
                    self._response_waiters[correlation_id].set()
        
        self.subscribe(reply_channel, handle_response)
        
        try:
            # Send request
            self.publish(
                target_channel,
                payload,
                correlation_id=correlation_id,
                reply_to=reply_channel
            )
            
            # Wait for response
            if event.wait(timeout):
                with self._response_lock:
                    return self._responses.pop(correlation_id, None)
            return None
        finally:
            # Cleanup
            self.unsubscribe(reply_channel, handle_response)
            with self._response_lock:
                self._response_waiters.pop(correlation_id, None)
                self._responses.pop(correlation_id, None)
    
    def respond(
        self,
        original_message: IPCMessage,
        payload: Dict[str, Any]
    ) -> Optional[str]:
        """
        Send response to a request.
        
        Args:
            original_message: The request message to respond to
            payload: Response payload
        
        Returns:
            Response message ID or None if no reply channel
        """
        if not original_message.reply_to:
            return None
        
        return self.publish(
            original_message.reply_to,
            payload,
            correlation_id=original_message.correlation_id,
            priority=MessagePriority.HIGH
        )
    
    # =========================================================================
    # SHARED CONTEXT METHODS
    # =========================================================================
    
    def set_shared(
        self,
        key: str,
        value: Any,
        context_type: str = "json",
        expires_seconds: Optional[int] = None,
        notify: bool = True
    ) -> None:
        """
        Store value in shared context.
        
        Args:
            key: Context key
            value: Value to store (will be JSON serialized)
            context_type: Type hint for value
            expires_seconds: Optional expiration
            notify: Publish update notification
        """
        expires_at = None
        if expires_seconds:
            expires_at = datetime.fromtimestamp(
                time.time() + expires_seconds
            ).isoformat()
        
        serialized = json.dumps(value) if context_type == "json" else str(value)
        
        with self._locked_write():
            conn = self._get_conn()
            
            # Get current version
            row = conn.execute(
                "SELECT version FROM shared_context WHERE key = ?", (key,)
            ).fetchone()
            version = (row['version'] + 1) if row else 1
            
            conn.execute("""
                INSERT OR REPLACE INTO shared_context 
                (key, value, context_type, owner_id, version, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                key, serialized, context_type, self._process_id,
                version, datetime.now().isoformat(), expires_at
            ))
            conn.commit()
        
        # Push notification
        if notify:
            self.publish("_context_updates", {
                "key": key,
                "action": "set",
                "version": version,
                "owner_id": self._process_id
            }, priority=MessagePriority.LOW)
    
    def get_shared(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from shared context.
        
        Args:
            key: Context key
            default: Default value if not found
        
        Returns:
            Stored value or default
        """
        conn = self._get_conn()
        row = conn.execute("""
            SELECT value, context_type, expires_at
            FROM shared_context
            WHERE key = ?
        """, (key,)).fetchone()
        
        if not row:
            return default
        
        # Check expiration
        if row['expires_at']:
            if datetime.fromisoformat(row['expires_at']) < datetime.now():
                return default
        
        if row['context_type'] == "json":
            return json.loads(row['value'])
        return row['value']
    
    def delete_shared(self, key: str, notify: bool = True) -> bool:
        """Delete shared context entry."""
        with self._locked_write():
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM shared_context WHERE key = ?", (key,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        
        if deleted and notify:
            self.publish("_context_updates", {
                "key": key,
                "action": "delete",
                "owner_id": self._process_id
            }, priority=MessagePriority.LOW)
        
        return deleted
    
    def get_all_shared(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Get all shared context, optionally filtered by key prefix."""
        conn = self._get_conn()
        
        if prefix:
            rows = conn.execute("""
                SELECT key, value, context_type, expires_at
                FROM shared_context
                WHERE key LIKE ?
            """, (f"{prefix}%",)).fetchall()
        else:
            rows = conn.execute(
                "SELECT key, value, context_type, expires_at FROM shared_context"
            ).fetchall()
        
        result = {}
        now = datetime.now()
        
        for row in rows:
            if row['expires_at'] and datetime.fromisoformat(row['expires_at']) < now:
                continue
            
            if row['context_type'] == "json":
                result[row['key']] = json.loads(row['value'])
            else:
                result[row['key']] = row['value']
        
        return result
    
    # =========================================================================
    # CONTEXT SYNCHRONIZATION
    # =========================================================================
    
    def subscribe_context_updates(
        self,
        callback: Callable[[str, str, Any], None]
    ) -> None:
        """
        Subscribe to context changes (push-based sync).
        
        Args:
            callback: Function(key, action, data) called on updates
        """
        def handle_update(msg: IPCMessage):
            key = msg.payload.get("key")
            action = msg.payload.get("action")
            
            if action == "set":
                value = self.get_shared(key)
                callback(key, action, value)
            else:
                callback(key, action, None)
        
        self.subscribe("_context_updates", handle_update)
    
    def sync_context_to_peer(self, peer_id: str, keys: Optional[List[str]] = None) -> int:
        """
        Push context to another peer.
        
        Args:
            peer_id: Target peer identifier
            keys: Specific keys to sync (None = all)
        
        Returns:
            Number of items synced
        """
        context = self.get_all_shared()
        if keys:
            context = {k: v for k, v in context.items() if k in keys}
        
        self.publish(f"_sync_{peer_id}", {
            "action": "full_sync",
            "context": context,
            "source_id": self._process_id
        })
        
        return len(context)
    
    # =========================================================================
    # BACKGROUND POLLING
    # =========================================================================
    
    def _start_polling(self) -> None:
        """Start background polling for new messages."""
        if self._polling:
            return
        
        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="IPC-Poller"
        )
        self._poll_thread.start()
    
    def _stop_polling(self) -> None:
        """Stop background polling."""
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
    
    def _poll_loop(self) -> None:
        """Background loop to check for new messages."""
        while self._polling:
            try:
                self._check_new_messages()
                time.sleep(0.1)  # 100ms polling interval
            except Exception as e:
                print(f"[IPC] Polling error: {e}")
                time.sleep(1.0)
    
    def _check_new_messages(self) -> None:
        """Check for new messages and notify subscribers."""
        with self._subscription_lock:
            channels = list(self._subscriptions.keys())
        
        if not channels:
            return
        
        conn = self._get_conn()
        
        # Get new messages for subscribed channels
        placeholders = ",".join("?" * len(channels))
        rows = conn.execute(f"""
            SELECT id, message_id, channel, sender_id, payload, priority,
                   correlation_id, reply_to, timestamp, expires_at
            FROM messages
            WHERE channel IN ({placeholders})
              AND id > ?
              AND processed = 0
              AND sender_id != ?
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY id ASC
            LIMIT 50
        """, (*channels, self._last_message_id, self._process_id, 
              datetime.now().isoformat())).fetchall()
        
        for row in rows:
            self._last_message_id = max(self._last_message_id, row['id'])
            
            msg = IPCMessage(
                id=row['message_id'],
                channel=row['channel'],
                sender_id=row['sender_id'],
                payload=json.loads(row['payload']),
                priority=row['priority'],
                correlation_id=row['correlation_id'],
                reply_to=row['reply_to'],
                timestamp=row['timestamp'],
                expires_at=row['expires_at']
            )
            
            self._notify_local_subscribers(row['channel'], msg)
            
            # Check for response waiters
            if msg.correlation_id:
                with self._response_lock:
                    if msg.correlation_id in self._response_waiters:
                        self._responses[msg.correlation_id] = msg
                        self._response_waiters[msg.correlation_id].set()
    
    # =========================================================================
    # CLEANUP METHODS
    # =========================================================================
    
    def cleanup_expired(self) -> int:
        """Remove expired messages and context. Returns count removed."""
        with self._locked_write():
            conn = self._get_conn()
            now = datetime.now().isoformat()
            
            # Remove expired messages
            cursor = conn.execute(
                "DELETE FROM messages WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            msg_count = cursor.rowcount
            
            # Remove expired context
            cursor = conn.execute(
                "DELETE FROM shared_context WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            ctx_count = cursor.rowcount
            
            conn.commit()
        
        return msg_count + ctx_count
    
    def cleanup_old_messages(self, max_age_hours: int = 24) -> int:
        """Remove old processed messages."""
        cutoff = time.time() - (max_age_hours * 3600)
        
        with self._locked_write():
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM messages WHERE processed = 1 AND created_at < ?",
                (cutoff,)
            )
            conn.commit()
        
        return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        conn = self._get_conn()
        
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE processed = 0"
        ).fetchone()[0]
        
        ctx_count = conn.execute(
            "SELECT COUNT(*) FROM shared_context"
        ).fetchone()[0]
        
        sub_count = len(self._subscriptions)
        
        return {
            "pending_messages": msg_count,
            "shared_context_keys": ctx_count,
            "active_subscriptions": sub_count,
            "process_id": self._process_id,
            "polling_active": self._polling
        }
    
    def close(self) -> None:
        """Clean shutdown."""
        self._stop_polling()
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()


# =============================================================================
# GLOBAL BROKER INSTANCE
# =============================================================================

_global_broker: Optional[IPCBroker] = None
_broker_lock = threading.Lock()


def get_broker(db_path: str = ".runtime/data/ipc_messages.db") -> IPCBroker:
    """Get the global IPC broker instance (singleton)."""
    global _global_broker
    
    with _broker_lock:
        if _global_broker is None:
            _global_broker = IPCBroker(db_path)
        return _global_broker


def reset_broker() -> None:
    """Reset the global broker (for testing)."""
    global _global_broker
    
    with _broker_lock:
        if _global_broker:
            _global_broker.close()
            _global_broker = None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("\n" + "="*60)
    print("IPC BROKER TEST")
    print("="*60 + "\n")
    
    # Use temp database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    broker = IPCBroker(db_path)
    
    # Test pub/sub
    print("Testing pub/sub...")
    received = []
    broker.subscribe("test_channel", lambda msg: received.append(msg))
    broker.publish("test_channel", {"test": "data"})
    time.sleep(0.2)
    print(f"  Received {len(received)} messages")
    
    # Test shared context
    print("\nTesting shared context...")
    broker.set_shared("target_ip", "10.0.0.1")
    broker.set_shared("scan_results", {"ports": [22, 80, 443]})
    ip = broker.get_shared("target_ip")
    print(f"  target_ip: {ip}")
    all_ctx = broker.get_all_shared()
    print(f"  All context: {all_ctx}")
    
    # Test request/response (in same process)
    print("\nTesting request/response...")
    def responder(msg: IPCMessage):
        if msg.reply_to:
            broker.respond(msg, {"status": "ok", "echo": msg.payload})
    
    broker.subscribe("rpc_test", responder)
    response = broker.request("rpc_test", {"action": "ping"}, timeout=2.0)
    print(f"  Response: {response.payload if response else 'TIMEOUT'}")
    
    # Stats
    print(f"\nStats: {broker.get_stats()}")
    
    # Cleanup
    broker.close()
    os.unlink(db_path)
    
    print("\n" + "="*60)
    print("IPC Broker tests completed!")
    print("="*60)
