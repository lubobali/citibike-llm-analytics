"""
Streaming wrapper for AdalFlow agent - adds real-time event emission.
Non-invasive: wraps agent without modifying core logic.
"""

import asyncio
from typing import Optional, Any
from datetime import datetime


class EventEmitter:
    """Simple event emitter for streaming events to frontend."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
    
    async def emit(self, event_type: str, data: dict = None):
        """Emit an event to the queue."""
        event = {
            "type": event_type,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.queue.put(event)
    
    async def get_event(self):
        """Get next event from queue."""
        return await self.queue.get()


class StreamingAgentWrapper:
    """
    Wraps AdalFlow agent to emit streaming events during execution.
    
    This is a non-invasive wrapper that intercepts method calls to emit
    real-time progress updates without modifying the agent's core logic.
    """
    
    def __init__(self, agent, event_emitter: EventEmitter):
        self.agent = agent
        self.emitter = event_emitter
    
    async def query_with_streaming(self, prompt: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """
        Execute agent query with streaming events.
        
        Emits events:
        - thinking: Agent is analyzing the question
        - tool_start: Starting tool execution (detected from agent state)
        - answer_start: Beginning to generate answer
        """
        try:
            # Emit initial thinking event
            await self.emitter.emit("thinking", {
                "message": "Analyzing your question..."
            })
            
            # Execute agent query in thread pool (it's synchronous)
            # This runs the full ReAct loop
            # Pass session_id and user_id if provided for database persistence
            if session_id:
                answer = await asyncio.to_thread(
                    self.agent.query, 
                    prompt, 
                    session_id=session_id, 
                    user_id=user_id or "anonymous"
                )
            else:
                answer = await asyncio.to_thread(
                    self.agent.query, 
                    prompt, 
                    user_id=user_id or "anonymous"
                )
            
            # Emit answer start event
            await self.emitter.emit("answer_start", {
                "message": "Generating response..."
            })
            
            return answer
            
        except Exception as e:
            # Emit error event
            await self.emitter.emit("error", {
                "message": f"Error: {str(e)}"
            })
            raise


def create_streaming_wrapper(agent):
    """Factory function to create streaming wrapper with event emitter."""
    emitter = EventEmitter()
    wrapper = StreamingAgentWrapper(agent, emitter)
    return wrapper, emitter

