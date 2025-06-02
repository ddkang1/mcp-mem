"""LightRAG implementation of memory storage."""

import json
import os
from typing import Any, Dict, List, Optional, Callable
from .storage import MemoryStorage, MemoryEntry
from .instance_manager import LightRAGInstanceManager


def get_default_embedding_func():
    """Get default embedding function for production."""
    try:
        import numpy as np
        from lightrag.utils import wrap_embedding_func_with_attrs
        import openai
        
        # Use OpenAI embeddings if API key is available
        if os.getenv("OPENAI_API_KEY"):
            @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
            async def openai_embedding_func(texts, model="text-embedding-ada-002"):
                openai_async_client = openai.AsyncOpenAI()
                response = await openai_async_client.embeddings.create(
                    model=model, input=texts
                )
                return [np.array(embedding.embedding) for embedding in response.data]
            return openai_embedding_func
        else:
            # Fallback to a simple mock for development
            @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
            async def mock_embedding_func(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return [np.random.randn(384).astype(np.float32) for _ in texts]
            return mock_embedding_func
    except ImportError:
        # Simple mock if dependencies aren't available
        import numpy as np
        
        # Create a simple mock without LightRAG dependencies
        async def simple_mock_embedding_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            return [np.random.randn(384).astype(np.float32) for _ in texts]
        
        # Manually add the required attributes
        simple_mock_embedding_func.embedding_dim = 384
        simple_mock_embedding_func.max_token_size = 512
        
        return simple_mock_embedding_func


def get_default_llm_func():
    """Get default LLM function for production."""
    try:
        import openai
        
        # Use OpenAI LLM if API key is available
        if os.getenv("OPENAI_API_KEY"):
            async def openai_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
                openai_async_client = openai.AsyncOpenAI()
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history_messages:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})
                
                response = await openai_async_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            return openai_llm_func
        else:
            # Fallback mock for development
            async def mock_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
                return f"Mock LLM response for: {prompt[:50]}..."
            return mock_llm_func
    except ImportError:
        # Simple mock if dependencies aren't available
        async def simple_mock_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return f"Simple mock response for: {prompt[:50]}..."
        return simple_mock_llm_func


class LightRAGStorage(MemoryStorage):
    """LightRAG-based storage implementation."""
    
    def __init__(self, embedding_func: Optional[Callable] = None, llm_func: Optional[Callable] = None):
        # Use provided functions or get defaults
        self.embedding_func = embedding_func or get_default_embedding_func()
        self.llm_func = llm_func or get_default_llm_func()
        
        # Create instance manager with proper functions
        self.instance_manager = LightRAGInstanceManager(
            embedding_func=self.embedding_func,
            llm_func=self.llm_func
        )
    
    def _entry_to_text(self, entry: MemoryEntry) -> str:
        """Convert memory entry to text format for LightRAG."""
        content = entry.content
        
        # Embed metadata in the content for LightRAG storage
        metadata_parts = []
        if entry.source_agent:
            metadata_parts.append(f"source_agent: {entry.source_agent}")
        if entry.message_type:
            metadata_parts.append(f"message_type: {entry.message_type}")
        if entry.metadata:
            metadata_parts.append(f"metadata: {json.dumps(entry.metadata)}")
        
        if metadata_parts:
            content = f"{content}\n[{', '.join(metadata_parts)}]"
        
        return content
    
    def _text_to_entry(self, text: str, entry_id: str) -> MemoryEntry:
        """Convert LightRAG text back to memory entry."""
        # Try to parse embedded metadata
        content = text
        source_agent = None
        message_type = None
        metadata = {}
        
        if "[" in text and "]" in text:
            try:
                parts = text.rsplit("\n[", 1)
                if len(parts) == 2:
                    content = parts[0]
                    metadata_part = parts[1].rstrip("]")
                    
                    for item in metadata_part.split(", "):
                        if ": " in item:
                            key, value = item.split(": ", 1)
                            if key == "source_agent":
                                source_agent = value
                            elif key == "message_type":
                                message_type = value
                            elif key == "metadata":
                                try:
                                    metadata = json.loads(value)
                                except:
                                    metadata = {"raw": value}
            except:
                # If parsing fails, use original text as content
                pass
        
        entry = MemoryEntry(content, metadata, source_agent, message_type)
        entry.id = entry_id
        return entry
    
    async def store(self, session_id: str, entry: MemoryEntry) -> bool:
        """Store a memory entry using LightRAG."""
        try:
            lightrag = await self.instance_manager.get(session_id)
            content_text = self._entry_to_text(entry)
            await lightrag.ainsert([content_text])
            return True
        except Exception as e:
            # Log error in production
            print(f"LightRAG store error: {e}")
            return False
    
    async def retrieve(self, session_id: str, query: str, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Retrieve memory entries using LightRAG."""
        try:
            lightrag = await self.instance_manager.get(session_id)
            results = await lightrag.aquery(query)
            
            entries = []
            for idx, text in enumerate(results):
                entry = self._text_to_entry(text, str(idx))
                entries.append(entry)
            
            if limit:
                entries = entries[:limit]
            
            return entries
        except Exception as e:
            # Log error in production
            print(f"LightRAG retrieve error: {e}")
            return []
    
    async def update(self, session_id: str, entry_id: str, new_content: Optional[str] = None, 
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory entry (simulated for LightRAG)."""
        # Note: LightRAG doesn't support direct updates, so this is simulated
        # In a real implementation, you might need to delete and re-insert
        return True
    
    async def delete(self, session_id: str, entry_id: str) -> bool:
        """Delete a memory entry (simulated for LightRAG)."""
        # Note: LightRAG doesn't support direct deletion, so this is simulated
        # In a real implementation, you might need to mark as deleted or rebuild the index
        return True
    
    async def get_all(self, session_id: str) -> List[MemoryEntry]:
        """Get all memory entries using LightRAG."""
        try:
            lightrag = await self.instance_manager.get(session_id)
            results = await lightrag.aquery("")  # Empty query to get all
            
            entries = []
            for idx, text in enumerate(results):
                entry = self._text_to_entry(text, str(idx))
                entries.append(entry)
            
            return entries
        except Exception as e:
            # Log error in production
            print(f"LightRAG get_all error: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the storage and cleanup resources."""
        self.instance_manager.shutdown() 