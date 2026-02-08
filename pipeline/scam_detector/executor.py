from typing import Optional
from llm.client import LLMClient
from utils import get_logger

logger = get_logger(__name__)

class LLMExecutor:
    """Executes prompts using the LLM client."""
   
    def __init__(self, model: Optional[str] = None) -> None:
        self.llm: LLMClient = LLMClient(model) if model else LLMClient()
        logger.info(f"Initialized LLMExecutor with model: {self.llm.model_name if hasattr(self.llm, 'model_name') else 'default'}")
   
    def execute(self, prompt: str) -> str:
        logger.info(f"Executing LLM with prompt length: {len(prompt)}")
        
        if len(prompt) > 500:
            logger.warning(f"Prompt is unusually long: {len(prompt)} characters")
        
        try:
            response = self.llm.call(prompt)
            logger.info(f"LLM execution successful, response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"LLM execution failed: {str(e)}")
            raise
