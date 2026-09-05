from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()


class BuildLLm:
    def __init__(self, model_url, model_name, api_key, temperature, frequency_penalty=None, presence_penalty=None, top_p=None, n=None, extra_body=None):
        self.model_url = model_url
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.llm = None
        self.top_p = top_p
        self.n = n
        self.extra_body = extra_body

    def build_llm(self):
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.model_url,
            temperature=self.temperature,
            extra_body=self.extra_body
        )
        return self.llm

    def set_paramaters(self, model_url, model_name, api_key, temperature):
        self.model_url = model_url
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature

    def get_llm(self):
        return self.llm


english_only_regex = (
    r"^[\t\n\r -~"           
    r"\u00A0-\u00FF"         
    r"\u0370-\u03FF"         
    r"\u2070-\u209F"         
    r"\u2190-\u21FF"         
    r"\u2200-\u22FF"         
    r"]*$"
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "llms_config.json")
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

# ✅ Get DeepSeek config from config file or environment
deepseek_config = cfg.get("deepseek", {})
deepseek_api_key = deepseek_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
deepseek_url = deepseek_config.get("model_url", "https://api.deepseek.com/v1")
deepseek_model = deepseek_config.get("model_name", "deepseek-chat")

print(f"🔑 DeepSeek API Key: {'✅ Set' if deepseek_api_key else '❌ Missing'}")
print(f"🌐 DeepSeek URL: {deepseek_url}")
print(f"🤖 DeepSeek Model: {deepseek_model}")

# For llama (if needed)
llama_llm_obj = BuildLLm(**cfg["llama"],
                         temperature=0)

# For qwen (if needed)
qwen_awq_32_llm_obj = BuildLLm(**cfg["qwen"],
                               temperature=0,
                               extra_body={"guided_regex": english_only_regex})

# ✅ Use DeepSeek as the judge (replaces GPT-4)
gpt_llm_obj = BuildLLm(
    model_url=deepseek_url,
    model_name=deepseek_model,
    api_key=deepseek_api_key,
    temperature=0
)

llama_llm = llama_llm_obj.build_llm()
qwen_llm = qwen_awq_32_llm_obj.build_llm()
gpt_llm = gpt_llm_obj.build_llm()

print(f"✅ Using LLM judge: {deepseek_model} (via {deepseek_url})")