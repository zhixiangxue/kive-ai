"""Request Field Mapping Bridge

Translates unified request fields (BaseMemoRequest) to backend-specific formats.
Maps context fields to mem0/cognee/graphiti parameters.
Handles messages conversion for backends that don't natively support it.
"""

from typing import Dict, Any, List

# Import BaseMemoRequest from models (avoid circular import by using TYPE_CHECKING)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models import BaseMemoRequest, AddMemoRequest, SearchMemoRequest


class RequestBridge:
    """Bridge for translating unified context to backend-specific formats
    
    Usage:
        from kive.models import AddMemoRequest
        
        bridge = RequestBridge()
        request = AddMemoRequest(
            app_id="my_app",
            user_id="U123",
            namespace="personal",
            text="Hello"
        )
        
        # Translate to different backends
        cognee_params = bridge.to_cognee_add(request)
        mem0_params = bridge.to_mem0_add(request)
        graphiti_params = bridge.to_graphiti_add(request)
    """
    
    @staticmethod
    def _convert_messages_to_text(messages: List[Dict[str, str]]) -> str:
        """Convert messages list to plain text
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            
        Returns:
            Plain text representation (role: content format)
        """
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def to_cognee_add(self, request: 'AddMemoRequest') -> Dict[str, Any]:
        """Translate to Cognee add() parameters
        
        Args:
            request: AddMemoRequest from API
            
        Returns:
            {
                "dataset_name": str,
                "user_id": str,  # CogneeAdapter will convert to User object
            }
            
        Note:
            Cognee only accepts text. If messages is provided, convert to text.
        """
        return {
            "dataset_name": request.namespace,
            "user_id": request.user_id,
        }
    
    def to_cognee_search(self, request: 'SearchMemoRequest') -> Dict[str, Any]:
        """Translate to Cognee search() parameters
        
        Args:
            request: SearchMemoRequest from API
            
        Returns:
            {
                "query_text": str,  # Cognee uses query_text, not query
                "top_k": int,  # Cognee uses top_k, not limit
                "datasets": str,
                "user_id": str,  # CogneeAdapter will convert to User object
                "session_id": str | None,
            }
            
        Note:
            Cognee search does NOT support: app_id, tenant_id, ai_id filtering.
            These fields are only stored in metadata for tracking.
        """
        result = {
            "query_text": request.query,  # Cognee API parameter name
            "top_k": request.limit,  # Cognee API parameter name
            "datasets": request.namespace,
            "user_id": request.user_id,
        }
        
        # Add session_id if not default
        if request.session_id and request.session_id != "default":
            result["session_id"] = request.session_id
        
        return result
    
    def to_mem0_add(self, request: 'AddMemoRequest') -> Dict[str, Any]:
        """Translate to Mem0 add() parameters
        
        Args:
            request: AddMemoRequest from API
            
        Returns:
            {
                "messages": str | list,  # Mem0 natively supports both formats
                "user_id": str,
                "agent_id": str | None,
                "run_id": str | None,
                "metadata": dict,  # Includes Kive context (_kive_*)
                "infer": bool,  # Default True
            }
        """
        # Priority: messages > text (Mem0 natively supports messages)
        content = request.messages if request.messages else (request.text or "")
        
        result = {
            "messages": content,  # Mem0 accepts both str and list[dict]
            "user_id": request.user_id,
            "infer": True,  # Enable intelligent inference by default
        }
        
        # Add agent_id if not default
        if request.ai_id != "default":
            result["agent_id"] = request.ai_id
        
        # Map session_id to run_id if not default
        if request.session_id != "default":
            result["run_id"] = request.session_id
        
        return result
    
    def to_mem0_search(self, request: 'SearchMemoRequest') -> Dict[str, Any]:
        """Translate to Mem0 search() parameters
        
        Args:
            request: SearchMemoRequest from API
            
        Returns:
            {
                "query": str,
                "limit": int,
                "user_id": str,
                "agent_id": str | None,
                "run_id": str | None,
            }
        """
        result = {
            "query": request.query,
            "limit": request.limit,
            "user_id": request.user_id,
        }
        
        # Add agent_id if not default
        if request.ai_id != "default":
            result["agent_id"] = request.ai_id
        
        # Map session_id to run_id if not default
        if request.session_id != "default":
            result["run_id"] = request.session_id
        
        return result
    
    def to_graphiti_add(self, request: 'AddMemoRequest') -> Dict[str, Any]:
        """Translate to Graphiti add_episode() parameters
        
        Args:
            request: AddMemoRequest from API
            
        Returns:
            {
                "episode_body": str,
                "name": str,
                "source_description": str,
                "reference_time": datetime,
                "group_id": str,  # namespace for partitioning
            }
            
        Note:
            Graphiti only accepts text. If messages is provided, convert to text.
        """
        from datetime import datetime
        
        # Convert messages to text if needed
        text_content = request.text
        if not text_content and request.messages:
            text_content = self._convert_messages_to_text(request.messages)
        
        return {
            "episode_body": text_content or "",
            "name": f"{request.namespace}_{request.user_id}",
            "source_description": request.namespace,
            "reference_time": datetime.now(),
            "group_id": request.namespace,  # Simple mapping: namespace → group_id
        }
    
    def to_graphiti_search(self, request: 'SearchMemoRequest') -> Dict[str, Any]:
        """Translate to Graphiti search() parameters
        
        Args:
            request: SearchMemoRequest from API
            
        Returns:
            {
                "query": str,
                "num_results": int,  # Graphiti uses num_results, not limit
                "group_ids": list[str] | None,
            }
            
        Note:
            Graphiti search does NOT support: app_id, tenant_id, ai_id, session_id filtering.
            These fields are only stored in metadata for tracking.
        """
        return {
            "query": request.query,
            "num_results": request.limit,  # Graphiti API parameter name
            "group_ids": [request.namespace] if request.namespace != "default" else None,
        }
    
    def get_external_metadata(self, request: 'BaseMemoRequest') -> Dict[str, Any]:
        """Get metadata fields that should be stored externally (in Kive layer)
        
        These fields are stored in Memo.metadata for tracking purposes.
        They allow reconstruction of the original context.
        
        Args:
            request: Unified request from API
            
        Returns:
            Dict containing all context fields with _kive_ prefix
        """
        metadata = {
            "_kive_app_id": request.app_id,
            "_kive_user_id": request.user_id,
            "_kive_namespace": request.namespace,
            "_kive_ai_id": request.ai_id,
            "_kive_tenant_id": request.tenant_id,
            "_kive_session_id": request.session_id,
        }
        
        # Merge with user-provided metadata
        if request.metadata:
            metadata.update(request.metadata)
        
        return metadata
