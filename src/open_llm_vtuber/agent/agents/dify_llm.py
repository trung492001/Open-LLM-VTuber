from typing import AsyncIterator, List, Dict, Any, Callable, Optional
from loguru import logger
import aiohttp
import json

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput
from ..input_types import BatchInput, TextSource, ImageSource
from ...chat_history_manager import get_history, get_metadata, update_metadate
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)


class DifyLLMAgent(AgentInterface):
    """
    Agent implementation using Dify.AI's API for chat completion.
    Implements text-based responses with sentence processing pipeline.
    """

    AGENT_TYPE = "dify_llm_agent"

    _system: str = """You are an error message repeater. 
        Your job is repeating this error message: 
        'No system prompt set. Please set a system prompt'. 
        Don't say anything else.
        """

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        live2d_model=None,
        tts_preprocessor_config=None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
    ):
        """
        Initialize the Dify agent with API credentials and configuration

        Args:
            api_endpoint: str - The Dify API endpoint URL
            api_key: str - The Dify API key
            live2d_model: Live2dModel - Model for expression extraction
            tts_preprocessor_config: TTSPreprocessorConfig - Configuration for TTS preprocessing
            faster_first_response: bool - Whether to enable faster first response
            segment_method: str - Method for sentence segmentation
        """
        super().__init__()
        self._api_endpoint = api_endpoint.rstrip('/')
        self._api_key = api_key
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        
        # Initialize aiohttp session
        self._session: Optional[aiohttp.ClientSession] = None
        self._conversation_id: Optional[str] = None
        self._current_conf_uid = None
        self._current_history_uid = None
        
        self.chat = self._chat_function_factory(self._dify_chat_completion)
        logger.info("DifyLLMAgent initialized.")

    async def _ensure_session(self):
        """Ensure an active session exists"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
            logger.debug("Created new aiohttp session")

    def _add_message(self, message: str, role: str):
        """Add a message to the memory"""
        self._memory.append(
            {
                "role": role,
                "content": message,
            }
        )

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load the memory from chat history and restore conversation if possible"""
        self._current_conf_uid = conf_uid
        self._current_history_uid = history_uid

        metadata = get_metadata(conf_uid, history_uid)
        
        agent_type = metadata.get("agent_type")
        if agent_type and agent_type != self.AGENT_TYPE:
            logger.warning(
                f"Incompatible agent type in history: {agent_type}. "
                f"Expected: {self.AGENT_TYPE} or empty. Memory will not be set."
            )
            self._conversation_id = None
            return

        self._conversation_id = metadata.get("conversation_id")
        if self._conversation_id:
            logger.info(f"Restored conversation ID from metadata: {self._conversation_id}")
        
        messages = get_history(conf_uid, history_uid)
        self._memory = [{"role": "system", "content": self._system}]

        for msg in messages:
            self._memory.append(
                {
                    "role": "user" if msg["role"] == "human" else "assistant",
                    "content": msg["content"],
                }
            )

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption"""
        if self._memory[-1]["role"] == "assistant":
            self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )
        self._memory.append(
            {
                "role": "system",
                "content": "[Interrupted by user]",
            }
        )

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format BatchInput into a prompt string"""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        if input_data.images:
            message_parts.append("\nImages in this message:")
            for i, img_data in enumerate(input_data.images, 1):
                source_desc = {
                    ImageSource.CAMERA: "captured from camera",
                    ImageSource.SCREEN: "screenshot",
                    ImageSource.CLIPBOARD: "from clipboard",
                    ImageSource.UPLOAD: "uploaded",
                }[img_data.source]
                message_parts.append(f"- Image {i} ({source_desc})")

        return "\n".join(message_parts)

    async def _dify_chat_completion(
        self, messages: List[Dict[str, Any]], system: str
    ) -> AsyncIterator[str]:
        """
        Perform chat completion using Dify's API

        Args:
            messages: List[Dict[str, Any]] - The conversation history
            system: str - System prompt

        Returns:
            AsyncIterator[str] - Token stream from Dify API
        """
        try:
            await self._ensure_session()
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",  # Explicitly request SSE
            }

            # Convert messages to Dify format
            dify_messages = []
            for msg in messages:
                if msg["role"] != "system":  # Skip system messages as they're handled differently
                    if isinstance(msg["content"], list):  # Handle rich content (images)
                        # For now, just extract text content
                        text_parts = [
                            content["text"]
                            for content in msg["content"]
                            if content["type"] == "text"
                        ]
                        content = " ".join(text_parts)
                    else:
                        content = msg["content"]
                    
                    dify_messages.append({
                        "role": msg["role"],
                        "content": content
                    })

            async with self._session.post(
                f"{self._api_endpoint}/chat-messages",
                headers=headers,
                json={
                    "inputs": {},
                    "query": dify_messages[-1]["content"],  # Latest user message
                    "response_mode": "streaming",
                    "conversation_id": self._conversation_id,
                    "user": "vtuber",
                },
                timeout=30  # Add timeout to prevent hanging
            ) as response:
                buffer = ""
                async for chunk in response.content.iter_chunks():
                    chunk_data = chunk[0].decode('utf-8') if chunk[0] else ''
                    buffer += chunk_data
                    
                    # Split buffer into individual SSE messages
                    while '\n\n' in buffer:
                        message, buffer = buffer.split('\n\n', 1)
                        # Process each line in the SSE message
                        for line in message.split('\n'):
                            if line.startswith('data: '):
                                data = line[6:].strip()  # Remove 'data: ' prefix
                                if not data:
                                    continue
                                try:
                                    json_data = json.loads(data)
                                    event_type = json_data.get("event")
                                    
                                    if event_type == "message":
                                        text = json_data.get("answer", "")
                                        if text:
                                            yield text
                                    elif event_type == "workflow_finished":
                                        # Handle end of conversation
                                        if json_data.get("data", {}).get("conversation_id"):
                                            self._conversation_id = json_data["data"]["conversation_id"]
                                            if self._current_history_uid:
                                                update_metadate(
                                                    self._current_conf_uid,
                                                    self._current_history_uid,
                                                    {
                                                        "conversation_id": self._conversation_id,
                                                        "agent_type": self.AGENT_TYPE
                                                    },
                                                )
                                                logger.info(f"Updated conversation ID: {self._conversation_id}")
                                
                                except json.JSONDecodeError as je:
                                    logger.error(f"JSON decode error: {je}")
                                    logger.error(f"Problematic data: {data}")
                                    continue

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            raise

    async def __del__(self):
        """Cleanup resources"""
        if self._session:
            await self._session.close()

    def _chat_function_factory(
        self, chat_func: Callable[[List[Dict[str, Any]], str], AsyncIterator[str]]
    ) -> Callable[..., AsyncIterator[SentenceOutput]]:
        """
        Create the chat pipeline with transformers

        The pipeline:
        LLM tokens -> sentence_divider -> actions_extractor -> display_processor -> tts_filter
        """

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(input_data: BatchInput) -> AsyncIterator[str]:
            """
            Chat implementation with memory and processing pipeline

            Args:
                input_data: BatchInput

            Returns:
                AsyncIterator[str] - Token stream from LLM
            """
            text_prompt = self._to_text_prompt(input_data)
            self._add_message(text_prompt, "user")

            # Get token stream from LLM
            token_stream = chat_func(self._memory, self._system)
            complete_response = ""

            async for token in token_stream:
                yield token
                complete_response += token

            # Store complete response
            self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(self, input_data: BatchInput) -> AsyncIterator[SentenceOutput]:
        """Placeholder chat method that will be replaced at runtime"""
        return self.chat(input_data)
