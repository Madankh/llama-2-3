from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
import math
from customGrpo import GRPO

SYSTEM_PROMPT = "Respond in following format : <thinking>{step by step reasoning}</thinking><answer>{number}</answer>"

# def prepare_dataset(dataset)->Dataset:
#     extract_hash_answer = (
#         lambda text : text.split("###")[1].strip() if "###" in text else None
#     )
    
#     def process_example(example:dict):
#         answer = extract_hash_answer(example["answer"])
#         if answer is None:
#             return None
        
#         return {
#             "prompt":[
#                 {"role":"system",
#                  "content":SYSTEM_PROMPT},
#                  {"role":"user",
#                   "content":example["quesion"]},
#             ],
#             "answer":answer
#         }
    
#     dataset = dataset.map(
#         process_example,
#         remove_columns = [
#             col for col in dataset.column_names if col not in ["prompt", "answer"]
#         ],
#     )
#     dataset = dataset.filter(lambda x:x is not None)
#     return dataset

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


model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
lora_rank = 32
model, tokenizer = FastLanguageModel.get_peft_model(
    model=model_name,
    r=lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407
)


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



