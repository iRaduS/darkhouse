import os
from . import AgentService
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM



class HuggingFaceAgentService(AgentService):
    DEFAULT_LANGUAGE_MODEL = os.getenv('LANGUAGE_MODEL_NAME', 'OpenLLM-Ro/RoGemma2-9b-Instruct-DPO-4Bit-BB')
    def __init__(self,
                 model_name: Optional[str] = DEFAULT_LANGUAGE_MODEL,
                 max_tokens_generated: Optional[int] = 128
                 ):
        self.max_tokens_generated = max_tokens_generated
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)

    @property
    def agent_instance(self):
        return self._model

    def agent_generate_content(self, requirement: str):
        prompt = self._tokenizer.apply_chat_template([{"role": "user", "content": requirement}], tokenize=False, system_message="")
        inputs = self._tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        outputs = self._model.generate(input_ids=inputs, max_new_tokens=self.max_tokens_generated)
        return self._tokenizer.decode(outputs[0])