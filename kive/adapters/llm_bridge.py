"""LLM Configuration Bridge

Translates unified LLM configuration to backend-specific formats.
Encapsulates all the complexity of different backends (Cognee, Graphiti, Mem0).
"""

from enum import Enum
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass


class LLMProvider(str, Enum):
    """Unified LLM service provider"""
    
    # Major cloud providers
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    
    # Chinese cloud providers
    BAILIAN = "bailian"          # Alibaba Bailian (DashScope)
    MOONSHOT = "moonshot"        # Moonshot AI (Kimi)
    DEEPSEEK = "deepseek"        # DeepSeek
    ZHIPU = "zhipu"              # Zhipu AI (ChatGLM)
    DOUBAO = "doubao"            # ByteDance Doubao
    TENCENT = "tencent"          # Tencent Hunyuan
    
    # Self-hosted
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    
    # Generic fallback
    OPENAI_COMPATIBLE = "openai-compatible"


# Type alias for LLM provider parameter (supports both enum and string)
LLMProviderType = Literal[
    "openai", "azure", "anthropic", "google", "groq",
    "bailian", "deepseek", "ollama", "lmstudio", "vllm", "openai-compatible"
]


@dataclass
class UnifiedLLMConfig:
    """Unified LLM configuration (kive standard)
    
    This is the standard format used across all adapters.
    """
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192


class LLMConfigBridge:
    """Bridge for translating unified config to backend-specific formats
    
    Usage:
        bridge = LLMConfigBridge()
        config = UnifiedLLMConfig(provider=LLMProvider.BAILIAN, model="qwen-plus", ...)
        
        # Translate to different backends
        cognee_config = bridge.to_cognee(config)
        graphiti_config = bridge.to_graphiti(config)
        mem0_config = bridge.to_mem0(config)
    """
    
    # Default base URLs for convenience (optional)
    DEFAULT_BASE_URLS = {
        LLMProvider.BAILIAN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        LLMProvider.MOONSHOT: "https://api.moonshot.cn/v1",
        LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1",
        LLMProvider.ZHIPU: "https://open.bigmodel.cn/api/paas/v4",
        LLMProvider.OLLAMA: "http://localhost:11434/v1",
    }
    
    def to_cognee(self, config: UnifiedLLMConfig) -> Dict[str, Any]:
        """Translate to Cognee format
        
        Cognee uses:
        - Native providers: openai, anthropic, gemini, ollama
        - Custom provider: for OpenAI-compatible endpoints
        
        Returns:
            Dict with keys: llm_provider, llm_model, llm_api_key, llm_endpoint
        """
        # Cognee's native providers
        cognee_native = {
            LLMProvider.OPENAI: "openai",
            LLMProvider.ANTHROPIC: "anthropic",
            LLMProvider.GOOGLE: "gemini",
            LLMProvider.OLLAMA: "ollama",
        }
        
        base_url = config.base_url or self.DEFAULT_BASE_URLS.get(config.provider)
        
        if config.provider in cognee_native:
            # Native provider, use directly
            return {
                "llm_provider": cognee_native[config.provider],
                "llm_model": config.model,
                "llm_api_key": config.api_key,
                "llm_endpoint": base_url,
            }
        else:
            # OpenAI-compatible, use "custom" provider
            return {
                "llm_provider": "custom",
                "llm_model": f"openai/{config.model}",  # Cognee's custom format
                "llm_api_key": config.api_key,
                "llm_endpoint": base_url,
            }
    
    def to_mem0(self, config: UnifiedLLMConfig) -> Dict[str, Any]:
        """Translate to Mem0 format
        
        Mem0 uses:
        - Provider name: openai, anthropic, gemini, ollama, deepseek, etc.
        - Provider-specific base_url field: {provider}_base_url
        
        Returns:
            Dict with keys: provider, config (containing model, api_key, base_url)
        """
        # Map kive provider to mem0 provider
        mem0_provider_map = {
            LLMProvider.OPENAI: "openai",
            LLMProvider.AZURE: "azure_openai",
            LLMProvider.ANTHROPIC: "anthropic",
            LLMProvider.GOOGLE: "gemini",
            LLMProvider.OLLAMA: "ollama",
            LLMProvider.GROQ: "groq",
            LLMProvider.DEEPSEEK: "deepseek",  # Mem0 has native deepseek support
            # Others use OpenAI-compatible
            LLMProvider.BAILIAN: "openai",
            LLMProvider.MOONSHOT: "openai",
            LLMProvider.ZHIPU: "openai",
            LLMProvider.DOUBAO: "openai",
            LLMProvider.OPENAI_COMPATIBLE: "openai",
        }
        
        mem0_provider = mem0_provider_map[config.provider]
        
        result = {
            "provider": mem0_provider,
            "config": {
                "model": config.model,
            }
        }
        
        # Add API key if provided
        if config.api_key:
            result["config"]["api_key"] = config.api_key
        
        # Add base_url with provider-specific field name
        base_url = config.base_url or self.DEFAULT_BASE_URLS.get(config.provider)
        if base_url:
            # Mem0's field naming: {provider}_base_url
            base_url_field = f"{mem0_provider}_base_url"
            result["config"][base_url_field] = base_url
        
        return result
    
    def to_graphiti(self, config: UnifiedLLMConfig):
        """Translate to Graphiti format and create LLM client instance
        
        Graphiti uses different Client classes:
        - OpenAIGenericClient (for OpenAI and compatible APIs)
        - AnthropicClient
        - GeminiClient
        - Custom extensions (e.g., BailianLLMClient)
        
        Returns:
            LLM client instance
        """
        # Import here to avoid circular dependency
        from graphiti_core.llm_client.config import LLMConfig
        
        base_url = config.base_url or self.DEFAULT_BASE_URLS.get(config.provider)
        
        # Create LLMConfig
        llm_config = LLMConfig(
            api_key=config.api_key or "placeholder",
            model=config.model,
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        # Determine which Client class to use and create instance
        if config.provider == LLMProvider.BAILIAN:
            # Our custom extension
            from .extensions.graphiti.llm_clients import BailianLLMClient
            return BailianLLMClient(config=llm_config)
        
        elif config.provider == LLMProvider.ANTHROPIC:
            from graphiti_core.llm_client.anthropic_client import AnthropicClient
            return AnthropicClient(config=llm_config)
        
        elif config.provider == LLMProvider.GOOGLE:
            from graphiti_core.llm_client.gemini_client import GeminiClient
            return GeminiClient(config=llm_config)
        
        elif config.provider == LLMProvider.GROQ:
            from graphiti_core.llm_client.groq_client import GroqClient
            return GroqClient(config=llm_config)
        
        else:  # OpenAIGenericClient (default for OPENAI, AZURE, MOONSHOT, DEEPSEEK, etc.)
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
            return OpenAIGenericClient(config=llm_config)
