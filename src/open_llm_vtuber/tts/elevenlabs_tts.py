import asyncio
from typing import Optional, Union
import tempfile
import os

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from loguru import logger

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """ElevenLabs TTS engine implementation."""

    def __init__(
        self, 
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ):
        """
        Initialize ElevenLabs TTS engine.

        Args:
            config: ElevenLabs TTS configuration
        """
        self.client = ElevenLabs(api_key=api_key)
        # Get voice and model objects
        self.voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost
        )
        self.voice_id = voice_id
        self.model_id = model_id

    def generate_audio(
        self,
        text: str,
        output_file: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to convert to speech
            output_file: Optional output file path
            language: Optional language code (not used for ElevenLabs)

        Returns:
            Path to the generated audio file or audio data as bytes
        """
        try:
            # Run the CPU-intensive generation in a thread pool
            audio = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                voice_settings=self.voice_settings,
                model_id=self.model_id,
                output_format="mp3_44100_128"
            )
            
            if output_file:
                # Save to specified output file
                with open(output_file, 'wb') as f:
                    for chunk in audio:
                        if chunk:
                            f.write(chunk)
                return output_file
            else:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(audio)
                    return temp_path

        except Exception as e:
            logger.error(f"Error in ElevenLabs TTS: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        pass  # No cleanup needed for ElevenLabs TTS 