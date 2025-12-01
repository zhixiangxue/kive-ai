"""Bailian LLM Client for Graphiti

Extension: Alibaba Cloud Bailian (DashScope) LLM client implementation

Issue:
    Graphiti uses OpenAI's Structured Output (json_schema response format),
    but Bailian's OpenAI-compatible API only supports basic json_object mode.

Solution:
    - Override _generate_response to use json_object instead of json_schema
    - Inject JSON schema into the prompt explicitly
    - Ensure prompt contains "json" keyword (required by Bailian)

Status: Production-ready ✅
Upstream Issue: N/A (API limitation, not a bug)

Usage:
    from kive.server.adapters.extensions.graphiti.llm_clients import BailianLLMClient
    
    llm_client = BailianLLMClient(config=llm_config)
"""

import json
import logging
import typing
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


class BailianLLMClient(OpenAIGenericClient):
    """Bailian-compatible LLM client for Graphiti
    
    Extends Graphiti's OpenAIGenericClient to work with Alibaba Cloud Bailian (DashScope),
    which has different JSON response formatting requirements:
    
    Differences from OpenAI:
        1. No support for json_schema (OpenAI Structured Output)
        2. Only supports json_object mode
        3. Requires the word "json" in the prompt when using json_object
    
    Implementation:
        - Overrides _generate_response to use json_object response format
        - Injects JSON schema into the user prompt
        - Adds explicit instructions for JSON output
    """
    
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate LLM response with Bailian-compatible JSON formatting
        
        Args:
            messages: List of conversation messages
            response_model: Pydantic model for structured output (optional)
            max_tokens: Maximum tokens in response
            model_size: Model size hint (not used currently)
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            ValueError: If LLM response is not valid JSON
            RateLimitError: If rate limit is exceeded
        """
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        
        try:
            # CRITICAL: Bailian requires using json_object instead of json_schema
            # Also need to ensure prompt contains the word "json"
            response_format: dict[str, Any] = {'type': 'json_object'}
            
            # If response_model is provided, add schema to the prompt instead of response_format
            if response_model is not None:
                json_schema = response_model.model_json_schema()
                schema_str = json.dumps(json_schema, indent=2, ensure_ascii=False)
                
                # Add explicit JSON format instruction to the last message
                # This ensures Bailian knows to output JSON
                json_instruction = (
                    f"\n\n**IMPORTANT: You must respond with a valid JSON object** "
                    f"that matches this exact schema:\n\n```json\n{schema_str}\n```\n\n"
                    f"Do not include any text outside the JSON object. "
                    f"Ensure all required fields are present and correctly typed."
                )
                
                # Append to the last user message
                if openai_messages and openai_messages[-1]['role'] == 'user':
                    openai_messages[-1]['content'] += json_instruction
                else:
                    # Fallback: add as new user message
                    openai_messages.append({'role': 'user', 'content': json_instruction})
            
            response = await self.client.chat.completions.create(
                model=self.model or 'qwen-plus',
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,  # Use json_object instead of json_schema
            )
            
            result = response.choices[0].message.content or ''
            
            # Parse and validate JSON response
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Raw response: {result}")
                raise ValueError(f"LLM did not return valid JSON: {e}")
            
            return parsed_result
            
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
