import requests
from loguru import logger
from .tts_interface import TTSInterface
import os


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://localhost:8000/api/styletts2",
        voice_description: str = "Trump",
    ):
        """
        Initialize StyleTTS2 TTS engine.
        
        Args:
            api_url: str
                URL of the StyleTTS2 API endpoint
            voice_description: str
                Voice description/style to use (e.g. "Trump")
        """
        self.api_url = api_url
        self.voice_description = voice_description
        self.new_audio_dir = "cache"
        self.file_extension = "wav"  # Adjust this based on your API's output format

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using StyleTTS2 API.
        
        Args:
            text: str
                The text to speak
            file_name_no_ext: str
                Name of the file without extension
        
        Returns:
            str: The path to the generated audio file
        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        # Prepare the request payload
        payload = {
            "sentence": text,
            "voice_description": self.voice_description
        }

        try:
            # Send POST request to the API
            response = requests.post(
                self.api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120
            )

            # Check if the request was successful
            if response.status_code == 200:
                # Save the audio content to a file
                with open(file_name, "wb") as audio_file:
                    audio_file.write(response.content)
                return file_name
            else:
                logger.error(f"API request failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return None