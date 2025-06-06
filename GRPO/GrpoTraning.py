from unsloth import FastLanguageModel
import torch
max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit = True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
    random_state=3407
)


import re
from datasets import load_dataset, Dataset
# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text:str)->str:
  answer = text.split("<answer>")[-1]
  answer = answer.split("</answer>")[0]
  return answer.strip()

def extract_hash_answer(text:str)->str| None:
  if '####' not in text:
    return None
  return text.split("####")[1].strip()

def get_gsm8k_questions(split="train")->Dataset:
  data = load_dataset('openai/gsm8k', 'main')[split]
  data = data.map(lambda x:{

      'prompt':[
          {'role':'system','content':SYSTEM_PROMPT},
          {'role': 'user', 'content' : x['question']}
      ],
      'answer':extract_hash_answer(x['answer'])
  })
  return data

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs)->list[float]:
  responses = [completion[0]['content'] for completion in completions]
  print(responses, "responses")
  q = prompts[0][-1]['content']
  print(q, "q")
  extracted_responses = [extract_xml_answer(r) for r in responses]
  print(extracted_responses, "extracted_responses")
  print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
  return [2.0 if r==a else 0.0 for r,a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwards)->list[float]:
  responses = [completion[0]['content'] for completion in completions]
  extracted_respose = [extract_xml_answer(r) for r in responses]
  return [0.5 if r.isdigit() else 0.0 for r in extracted_respose]

def strict_format_reward_func(complections, **kwards):
     """Reward function that checks if the completion has a specific format."""
     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
     responses = [complection[0]["content"] for complection in complections]
     matches = [re.match(pattern, r) for r in responses]
     return [0.5 if match else 0.0 match in matches]

def soft_format_reward_func(complections, **kwargs)->list[float]:
  """Reward function that checks if the completion has a specific format."""
  pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?<answer>"
  responses = [completion[0]["content"] for completion in complections]
  matches = [re.match(pattern, r) for r in responses]
  return [0.5 if match else 0.0 for match in matches]

def count_xml(text)->float:
  count = 0.0
  if text.count("<reasoning>\n") == 1:
    count += 0.125
  if text.count("\n</reasoning>") == 1:
    count += 0.125
  if text.count("<answer>\n") == 1:
    count += 0.125
    count -= len(text.split("\n</answer>")[-1])*0.001
  if text.count("\n</answer>"):
    count += 0.125
    count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
  return count


def xmlcount_reward_func(completions, **kwargs)->list[float]:
  contents = [completion[0]["content"] for completion in completions]
  return [count_xml(c) for c in contents]

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta = 0.9,
    adam_beta2 = 0.99
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 6,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "outputs"
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
    
)
trainer.train()
model.save_lora("grpo_base_model")
text = tokenizer.apply_chat_template([
    {"role" : "system", "content": SYSTEM_PROMPT},
    {"role" : "user", "content" : "Calculate pi"},
], tokenize=False, add_generation_prompt=True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_base_model"),
)[0].outputs[0].text

