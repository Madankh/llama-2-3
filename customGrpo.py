import torch
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time
import wandb
import datetime
from collections import defaultdict

class GRPO:
    def __init__(self,
                 model, 
                 ref_model,
                 tokenizer=None, 
                 group_size=8,
                 micro_group_size=2,
                 batch_size=1, 
                 dataset=None, 
                 max_iterations=1000,
                 reward_functions=None, 
                 log_wandb=False,
                 dtype=None,
                 lr=5e-6,
                 weight_decay=0.0,
                 beta=0.0,
                 epsilon=0.1):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        self.micro_group_size = micro_group_size
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = beta
        self.epsilon = epsilon
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        assert reward_functions if not None, "Must pass reward_functions"
        self.reward_functions:list = reward_functions
        self.using_lora = True if self.ref_model is None else False
        if self.using_lora:
            self.ref_model = model
        self.distributed = False
        self.log_wandb = log_wandb
        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="nanoGRPO")
        self.metrics = defaultdict(list)
        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)
    def get_per_token_logs(self, model, input_ids)->Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:,:-1,:]
        input_ids = input_ids[:,1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)
    
    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards,
                     std_rewards, loss_mask)->Tensor:
        policy_log_probs = self.get_per_token_logs(self.model, inputs)

        with (
            self.ref_model.disable_adapter()
            if self.using_lora
            else contextlib.nullcontext()
        ):
            ref_policy_log_prob = self.get_per_token_logps(self.ref_model, inputs)


        # advantage calculation
        advantage:Tensor = (reward - mean_rewards)/(std_rewards + 1e-6)
        advantage = advantage.reshape(-1,1)

        # KL divergene calculation
        log_rations = ref_policy_log_prob - policy_log_probs
        kld = torch.exp(log_rations) - log_rations - 1

        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())
        loss1 = policy_ratio*advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        loss = (loss * loss_mask).sum(dim=-1)/(loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1)/(loss_mask.sum(dim=-1) + 1e-6)
        loss += kld * self.beta
        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())
        return loss.mean()
    

    # def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
    #     policy_log_probs = self.get_per_token_logps(self.model, inputs)
        
    #     with (
    #         self.ref_model.disable_adapter()
    #         if self.using_lora  
    #         else contextlib.nullcontext()
    #     ):
    #         ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)


    #     # advantage calculation
    #     advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
    #     advantage = advantage.reshape(-1, 1)

    #     # kl divergence calculation
    #     log_ratios = ref_policy_log_probs - policy_log_probs
    #     kld = torch.exp(log_ratios) - log_ratios - 1

    #     policy_ratio = torch.exp(policy_log_probs-old_policy_log_probs.detach())

    #     loss1 = policy_ratio*advantage
    #     loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
    #     loss = -torch.min(loss1, loss2)
    #     loss = (loss * loss_mask).sum(dim=-1)/ (loss_mask.sum(dim=-1) + 1e-6)
    #     kld = (kld * loss_mask).sum(dim=-1)/ (loss_mask.sum(dim=-1) + 1e-6)
    #     loss += kld * self.beta
    #     if self.log_wandb:
    #         for _kd in kld:
    #             self.metrics["kld"].append(_kd.mean().item())
    #     return loss.mean()

    def sample_batch(self):
        if self.distributed:
            return self.distributed_sample_batch()
        inputs_texts = []
        samples = []
        for _ in  range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            inputs_texts.append(formatted)
        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']