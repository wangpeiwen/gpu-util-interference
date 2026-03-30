import os
os.environ['VLLM_USE_V1'] = '0'
from vllm import LLM
llm = LLM(model='/data/Qwen/Qwen2.5-7B-Instruct', dtype='float16', trust_remote_code=True, enforce_eager=True)
print('start_profile:', hasattr(llm, 'start_profile'))
print('llm_engine:', hasattr(llm, 'llm_engine'))
if hasattr(llm, 'llm_engine'):
    print([x for x in dir(llm.llm_engine) if 'prof' in x.lower()])
