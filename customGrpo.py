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
        # assert reward_functions if not None, "Must pass reward_functions"
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
    

    def sample_batch(self):
        # Check if running in distributed mode - if so, use specialized method for distributed processing
        if self.distributed:
            return self.distributed_sample_batch()
        
        # Initialize empty lists for storing formatted input texts and original samples
        inputs_texts = []
        samples = []
        
        # Loop to collect batch_size number of samples from the data loader
        for _ in range(self.batch_size):  # Note: Typo, should be self.batch_size
            # Get the next item from the data loader iterator
            item = next(self.data_loader_iter)
            
            # Store the complete item (contains prompt and likely other metadata) in samples list
            samples.append(item)
            
            # Extract just the prompt field from the item
            prompt = item["prompt"]
            
            # Format the prompt using the tokenizer's chat template (adds system/user markers)
            # tokenize=False means we get back text, not tokens yet
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            
            # Add the formatted prompt to our collection
            inputs_texts.append(formatted)
        
        # Convert all formatted texts to token IDs with padding to make them uniform length
        # return_tensors="pt" ensures we get PyTorch tensors back
        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        
        # Extract the token IDs and attention mask from the encoded result
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Store the length of the prompts - we'll need this to separate prompt from generation later
        prompt_length = input_ids.shape[1]
        
        # Repeat each input group_size times to generate multiple variations per prompt
        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        
        # Duplicate the samples list to match the repeated inputs
        # Note: Typo here, should be self.group_size not self.group*size
        samples = [sample for _ in range(self.group_size) for sample in samples]
        
        # Start timing the generation process
        start_time = time.time()
        
        # Set maximum number of new tokens to generate
        max_new_tokens = 512
        
        # Generate text completions using the model
        outputs = self.model.generate(
            # Move input_ids to the target device (CPU/GPU)
            input_ids.to(self.device),
            # Generate up to 512 new tokens (after the prompt)
            max_new_tokens=max_new_tokens,
            # Use temperature of 0.9 for some randomness in generation
            temperature=0.9,
            # Note: Commented out parameters below show other possible settings
            # min_new_tokens=512,
            # repetition_penalty=1.1,
        )
        
        # Stop timing and report generation time
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")
        
        # Convert token IDs back to text (keeping special tokens for complete output)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Calculate reward scores for each generated output
        rewards = self.compute_rewards(samples, decoded_outputs)
        
        # Create a mask tensor of the same shape as outputs, initially all False/0
        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)
        
        # Extract just the generated part (after the prompt)
        gen_tokens = outputs[:, prompt_length:]
        
        # Create a mask that's True for all generated tokens except padding tokens
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        
        # Update the loss mask to include only valid generated tokens
        loss_mask[:, prompt_length:] = valid_gen_mask
        
        # Return three key elements for training:
        # 1. Full token outputs (prompt + generation)
        # 2. Rewards tensor (converted to floating point with appropriate dtype)
        # 3. Loss mask shifted by one position (for next-token prediction)
        return outputs, torch.tensor(rewards, dtype=self.dtype).float(), loss_mask[:, 1:]
    
    def compute_rewards(self, samples, responces)->list:
        rewards = [[[] for _ in range(self.batch_size)] for _ in range(len(self.reward_functions))]

        for idx, (sample, resp) in enumerate(zip(samples, responces)):
            reward = 0
            for func_idx, func in enumerate(self.reward_functions):
                reward += func(sample, resp)
                rewards[func_idx][idx % self.batch_size].append(reward)
            rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)

            for func_idx, func in enumerate(self.reward_functions):
                rwds = rewards[func_idx].mean(dim=-1)
                for r in rwds:
                    self.metrics[f"reward_ {func.__name__}"].append(r.item())
            
            prompt_lengths = [[] for _ in range(self.batch_size)]
            for idx, sample in enumerate(samples):
                prompt_lengths[idx % self.batch_size].append(len(sample["prompt"]))
            
            for idx, pl in enumerate(prompt_lengths):
                self.metrics[f"prompt_length"].append(sum(pl)/len(pl))
            
            return rewards.sum(dim=0)
        
    def log_metrics(self):
        if self.log_wandb:
            idx = self.metrics["idx"][-1]-1
            metrics = {}
            for k, v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]
            wandb.log(metrics)
        
    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        start_time= time.perf_counter()
        while idx < max_iterations:
            x_batch_inputs , rewards , loss_mask = self.sample_batch()
            torch.cuda.empty_cache()
            
            # Reshape batch into [Batch_size, group_size, seq_len]
            # This separates the orignal prompts from their multiple variation
            batch_inputs = x_batch_inputs.reshape(self.batch_size, self.group_size, *x_batch_inputs.shape[1:])
            loss_mask = loss_mask.reshape(self.batch_size, self.group_size, *loss_mask.shape[1:]) 

            # Free up more gpu memory
            torch.cuda.empty_cache()

            # Move data to cpu to save gpu memory
            batch_inputs = batch_inputs.cpu()
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()  

            # Free up gpu memory again
            torch.cuda.empty_cache()

            # Initialize list to store  the old policy log probabilities
            pi_old = []

            for _, (b_inputs) in enumerate(batch_inputs):
                # Disable gradient calculation since we're just evaluating the old policy
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logs(self.model, b_inputs.to(self.device)).cpu()
                    # Free GPU memory
                    torch.cuda.empty_cache()

                    # Stoore these loog probabilites
                    pi_old.append(b_old_policy_log_probs)
                # Process each batch item individually
                for _, (b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask) in enumerate(zip(batch_inputs, pi_old, rewards, loss_mask)):
                    idx+=1

                    # Move rewards to the device (GPU)
                    reward = b_reward.to(self.device)

                    # Calculaate mean and std of rewards for normalization
                    mean_rewards = rewards.mean(dim=-1).unsqueeze(-1)
                    std_rewards = reward.std(dim=-1).unsqueeze(-1)

                    # Further split the groups into micro-groups to handle gpu memory constraints
                    # This is like micro-batching - processing smaller chunks at a time
                    g_inputs = b_inputs.reshape(b_inputs.shape[0]//self.micro_group_size, self.micro_group_size, *b_inputs.shape[1:]).cpu()
                    g_old_policy_log_prob = b_old_policy_log_probs.reshape(b_inputs.shape[0]//self.micro_group_size, self.micro_group_size, *b_old_policy_log_probs.shape[1:])
                    g_reward = b_reward.reshape(b_inputs.shape[0]//self.micro_group_size, self.micro_group_size, *b_reward.shape[1:]).cpu()
                    g_loss_mask = b_loss_mask.reshape(b_inputs.shape[0]//self.micro_group_size, self.micro_group_size, *b_loss_mask.shape[1:]).cpu()
                    group_losses = []

                    for inputs, old_policy_log_probs, reward, loss_mask in zip(g_inputs, g_old_policy_log_prob, g_reward, g_loss_mask):
                        inputs = inputs.to(self.device)
                        old_policy_log_probs = old_policy_log_probs.to(self.device)
                        reward = reward.to(self.device)
                        loss_mask = loss_mask.to(self.device)

                        loss = self.compute_loss(inputs,
                            old_policy_log_probs,
                            reward,
                            mean_rewards,
                            std_rewards,
                            loss_mask
                        )
                        group_losses.append(loss.item())
                        loss.backward()
                        torch.cuda.empty_cache()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.log_wandb:
                        self.metrics["idx"].append(idx)
                        self.metrics["total_reward"].append(reward.mean().item())
                        self.metrics["loss"].append(sum(group_losses)/len(group_losses))
                    torch.cuda.empty_cache()

                print(f"iter {idx} >>> reward : {reward.mean()}")
                print(f"Total time : {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
                self.log_metrics()

                                           
                










        
        

        



