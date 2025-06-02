"""Abstract storage interface for memory management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import datetime


class MemoryEntry:
    """A memory entry with content and metadata."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                 source_agent: Optional[str] = None, message_type: Optional[str] = None, 
                 timestamp: Optional[str] = None):
        self.content = content
        self.metadata = metadata or {}
        self.source_agent = source_agent
        self.message_type = message_type
        self.timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.id: Optional[str] = None  # Will be set by storage implementation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "content": self.content,
            "metadata": self.metadata,
        }
        if self.id is not None:
            result["id"] = self.id
        if self.source_agent:
            result["source_agent"] = self.source_agent
        if self.message_type:
            result["message_type"] = self.message_type
        if self.timestamp:
            result["timestamp"] = self.timestamp
        return result


class MemoryStorage(ABC):
    """Abstract base class for memory storage implementations."""
    
    @abstractmethod
    async def store(self, session_id: str, entry: MemoryEntry) -> bool:
        """Store a memory entry. Returns True if successful."""
        pass
    
    @abstractmethod
    async def retrieve(self, session_id: str, query: str, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Retrieve memory entries matching the query."""
        pass
    
    @abstractmethod
    async def update(self, session_id: str, entry_id: str, new_content: Optional[str] = None, 
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory entry. Returns True if successful."""
        pass
    
    @abstractmethod
    async def delete(self, session_id: str, entry_id: str) -> bool:
        """Delete a memory entry. Returns True if successful."""
        pass
    
    @abstractmethod
    async def get_all(self, session_id: str) -> List[MemoryEntry]:
        """Get all memory entries for a session."""
        pass


class InMemoryStorage(MemoryStorage):
    """Simple in-memory storage implementation for testing."""
    
    def __init__(self):
        self._storage: Dict[str, List[MemoryEntry]] = {}
        self._next_id = 0
    
    def _get_next_id(self) -> str:
        """Generate next unique ID."""
        self._next_id += 1
        return str(self._next_id)
    
    async def store(self, session_id: str, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        if session_id not in self._storage:
            self._storage[session_id] = []
        
        entry.id = self._get_next_id()
        self._storage[session_id].append(entry)
        return True
    
    async def retrieve(self, session_id: str, query: str, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Retrieve memory entries matching the query."""
        if session_id not in self._storage:
            return []
        
        # Simple text search in content
        results = []
        for entry in self._storage[session_id]:
            if query.lower() in entry.content.lower():
                results.append(entry)
        
        if limit:
            results = results[:limit]
        
        return results
    
    async def update(self, session_id: str, entry_id: str, new_content: Optional[str] = None, 
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory entry."""
        if session_id not in self._storage:
            return False
        
        for entry in self._storage[session_id]:
            if entry.id == entry_id:
                if new_content:
                    entry.content = new_content
                if new_metadata:
                    entry.metadata.update(new_metadata)
                return True
        
        return False
    
    async def delete(self, session_id: str, entry_id: str) -> bool:
        """Delete a memory entry."""
        if session_id not in self._storage:
            return False
        
        for i, entry in enumerate(self._storage[session_id]):
            if entry.id == entry_id:
                del self._storage[session_id][i]
                return True
        
        return False
    
    async def get_all(self, session_id: str) -> List[MemoryEntry]:
        """Get all memory entries for a session."""
        return self._storage.get(session_id, []) 