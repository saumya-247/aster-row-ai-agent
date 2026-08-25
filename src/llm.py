import logging
import os
from typing import Optional

try:
    from src.config import GEMINI_API_KEY, GEMINI_MODEL, AGENT_SYSTEM_PROMPT
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    AGENT_SYSTEM_PROMPT = ""

logger = logging.getLogger("AsterRowAgent.LLM")


class LLMProvider:
    """
    LLM Provider abstraction for Aster & Row Support Agent.
    Supports Google Gemini (via google.generativeai or google.genai).
    Gracefully falls back if API key is not configured, library is missing,
    or remote API calls encounter errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        self.system_instruction = system_instruction or AGENT_SYSTEM_PROMPT
        self._client = None
        self._provider_type = None

        self._initialize_client()

    def _initialize_client(self):
        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            logger.info("No valid GEMINI_API_KEY configured. LLM provider will operate in offline/fallback mode.")
            return

        # Attempt initializing with google.generativeai
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction if self.system_instruction else None
            )
            self._provider_type = "google-generativeai"
            logger.info(f"Gemini LLM Provider successfully initialized using model: {self.model_name}")
            return
        except Exception as e:
            logger.debug(f"google.generativeai initialization skipped/failed: {e}")

        # Attempt initializing with google.genai
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._provider_type = "google-genai"
            logger.info(f"Gemini LLM Provider initialized via google.genai Client: {self.model_name}")
            return
        except Exception as e:
            logger.debug(f"google.genai initialization skipped/failed: {e}")

    def is_available(self) -> bool:
        """Returns True if an LLM client is configured and ready."""
        return self._client is not None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1
    ) -> Optional[str]:
        """
        Generates a text completion from the LLM.
        Returns None if provider is not available or if the API call fails,
        enabling safe fallback.
        """
        if not self.is_available():
            return None

        try:
            if self._provider_type == "google-generativeai":
                import google.generativeai as genai
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                )
                response = self._client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                if response and response.text:
                    return response.text.strip()
            elif self._provider_type == "google-genai":
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}. Falling back to deterministic synthesizer.")
            return None

        return None
