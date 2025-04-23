import torch
import torch.nn.functional as F
import contextlib
import wandb
from torch import Tensor
import time
import datetime
from collections import defaultdict

## Import done

## starting GRPO 
class GRPO:
    def __init__(self,
                 model,
                 ref_model,
                 tokenizer=None,
                 group_size=8,
                 min_group_size=2,
                 batch_size = 1,
                 max_iterations=1000,
                 dataset=None,
                 reward_functions=None,
                 log_wandb = False,
                 dtype=None,
                 lr=5e-6,
                 weight_decay = 0.0,
                 beta = 0.0,
                 epsilon=0.1
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.micro_group_size = min_group_size
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.epsilon = epsilon
        assert reward_functions is not None, "Must pass reward functions"
        self.dtype = dtype if dtype is None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.reward_funtions:list = reward_functions
        self.using_lora = True if self.ref_model is None else False
        if  self.using_lora:
            self.ref_model=model
        self.distributed=False
        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="madancustomGRPO")
        self.metrics = defaultdict(list)
        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)

    def get_per_token_logs(self,model, input_ids)->Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:,:-1,:]
        inputs_ids = input_ids[:,1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, inputs_ids.unsqueeze(-1)).squeeze(-1)
    
    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask)->Tensor:
        policy_log_probs = self.get_per_token_logs(self.model, inputs)

        with (
            self.ref_model.disable_adapter()
            if self.using_lora 
            else contextlib.nullcontext()
        ):
            ref_policy_log_probs = self.get_per_token_logs(self.ref_model, inputs)

        # Advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)

        # KL divergencec calculation
        log_ration = ref_policy_log_probs - policy_log_probs
        kld = torch.exp(log_ration) - log_ration - 1
        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs)
        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1 , loss2)
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        loss += kld * self.beta

        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())
        return loss.mean()
    
    def compute_rewards(self, samples, responses)->list:
        rewards = [[[] for _ in range(self.batch_size)] for _ in range(len(self.reward_funtions))]

        for idx, (sample, resp) in enumerate(zip(samples, responses)):
            reward = 0
            for func_idx, func in enumerate(self.reward_funtions):
                reward += func(sample, resp)
                rewards[func_idx][idx%self.batch_size].append(reward)
        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)
        
        # print(f"rewards:{rewards.shape})
        for func_idx, func in enumerate(self.reward_funtions):
            rwds = rewards[func_idx].mean()
            for r in rwds:
                self.metrics[f"reward_{func.__name__}"].append(r.item())
        prompt_lengths = [[] for _ in range(self.batch_size)]
        for idx , sample in enumerate(samples):
            prompt_lengths[idx%self.batch_size].append(len(sample["prompt"]))
        
        for idx, pl in enumerate(prompt_lengths):
            self.metrics[f"prompt_length"].append(sum(pl)/len(pl))

        return rewards.sum(dim=0)
    
    def log_metrics(self):
        if self.log_wandb:
            idx = self.metrics["idx"][-1]-1
            metrics = {}
            for k , v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]
            wandb.log(metrics)


    def sample_batch(self):
        if self.distributed:
            return self.distributed_sample_batch()
        
        inputs_texts = []
        samples = []

        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenizer=False)
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]
        start_time = time.time()
        max_new_tokens = 512
        outputs = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens = max_new_tokens,
            temperature = 0.9
        )
        end_time = time.time()
        print(f"Time for generation : {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        rewards = self.compute_rewards(samples, decoded_outputs)

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:,prompt_length:]

        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:,prompt_length:] = valid_gen_mask

        return outputs , torch.tensor(rewards, dtype=self.dtype).float(), loss_mask[:,1:]
    
    def train(self,epochs=1, max_iterations=1000):
        idx = 0
        start_time = time.perf_counter()
        while idx < max_iterations:

            x_batch_inputs, rewards, loss_mask = self.sample_batch()
            torch.cuda.empty_cache()

            x_batch_inputs = x_batch_inputs.reshape(self.batch_size, self.group_size, x_batch_inputs.shape[1:])
            loss_mask = loss_mask.reshape(self.batch_size, self.group_size, loss_mask.shape[1:])
            torch.cuda.empty_cache()

            ## Offload to cpu to save vram
            batch_inputs = batch_inputs.cpu()
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()
            torch.cuda_empty_cache()

            pi_old = []

            for _ , (b_inputs) in enumerate(batch_inputs):
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logs(self.model, b_inputs.to(self.device)).cpu()
                    torch.cuda.empty_cache()
                    pi_old.append(b_old_policy_log_probs)

            for _, (b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask) in enumerate(zip(batch_inputs, pi_old, rewards, loss_mask)):
                idx +=1
                reward = b_reward.to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)

                g_inputs                =                b_inputs.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, b_inputs.shape[1:]).cpu()
                g_old_policy_log_probs  =  b_old_policy_log_probs.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, b_old_policy_log_probs.shape[1:]).cpu()
                g_reward =                               b_reward.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, b_reward.shape[1:]).cpu()
                g_loss_mask =                         b_loss_mask.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, b_loss_mask.shape[1:]).cpu()
                group_losses = []

                for inputs, old_policy_log_probs, reward, loss_mask in zip(g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask):
                    


                    
            

            








    
    

    