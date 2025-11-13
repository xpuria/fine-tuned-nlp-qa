"""Reinforcement Learning from Human Feedback (RLHF) trainer."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Reward model for RLHF."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use the last token's hidden state for reward prediction
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden_state)
        return reward


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer using PPO."""

    def __init__(
        self,
        model_name: str,
        reward_model: RewardModel,
        kl_coef: float = 0.2,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        learning_rate: float = 1e-5,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model name or path
            reward_model: Trained reward model
            kl_coef: KL divergence coefficient
            clip_range: PPO clipping range
            value_coef: Value function coefficient
            learning_rate: Learning rate
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.learning_rate = learning_rate
        self.max_length = max_length

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.reward_model = reward_model
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Value function (separate from policy)
        self.value_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.value_head = nn.Linear(self.value_model.config.hidden_size, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model.to(self.device)
        self.ref_model.to(self.device)
        self.reward_model.to(self.device)
        self.value_model.to(self.device)
        self.value_head.to(self.device)

        logger.info(f"Initialized RLHF trainer with model: {model_name}")
        logger.info(f"Using device: {self.device}")

    def generate_responses(self, prompts: List[str], num_samples: int = 4) -> List[str]:
        """Generate responses for given prompts."""
        self.policy_model.eval()
        responses = []

        for prompt in prompts:
            prompt_tokens = self.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.device)

            generated_responses = []
            for _ in range(num_samples):
                with torch.no_grad():
                    output = self.policy_model.generate(
                        **prompt_tokens,
                        max_length=self.max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(
                    output[0][len(prompt_tokens['input_ids'][0]):],
                    skip_special_tokens=True
                )
                generated_responses.append(response)

            responses.extend(generated_responses)

        self.policy_model.train()
        return responses

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards for prompt-response pairs."""
        rewards = []

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}\n\n{response}"
            tokens = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                reward = self.reward_model(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens['attention_mask']
                )
            rewards.append(reward.item())

        return torch.tensor(rewards, device=self.device)

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> torch.Tensor:
        """Compute generalized advantage estimates."""
        advantages = torch.zeros_like(rewards)

        last_gae_lam = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_values = 0
            else:
                next_values = values[step + 1]

            delta = rewards[step] + gamma * next_values - values[step]
            advantages[step] = last_gae_lam = delta + gamma * lam * last_gae_lam

        return advantages

    def ppo_step(
        self,
        old_log_probs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ):
        """Perform one PPO optimization step."""
        # Get new log probabilities
        outputs = self.policy_model(
            input_ids=states,
            attention_mask=(states != self.tokenizer.pad_token_id).long()
        )
        new_logits = outputs.logits

        # Compute new log probs for actions
        new_log_probs = torch.gather(
            torch.log_softmax(new_logits, dim=-1),
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value function loss
        values = self.value_head(outputs.hidden_states[-1][:, -1, :])
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        # KL divergence penalty
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=states,
                attention_mask=(states != self.tokenizer.pad_token_id).long()
            )
            ref_logits = ref_outputs.logits

        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(new_logits, dim=-1),
            torch.softmax(ref_logits, dim=-1)
        )

        total_loss = policy_loss + self.value_coef * value_loss + self.kl_coef * kl_loss

        return total_loss

    def train(
        self,
        prompts: List[str],
        num_iterations: int = 10,
        num_samples_per_prompt: int = 4,
        batch_size: int = 4,
        output_dir: str = "./rlhf_results"
    ):
        """
        Train using RLHF with PPO.

        Args:
            prompts: List of prompts for training
            num_iterations: Number of PPO iterations
            num_samples_per_prompt: Number of samples per prompt
            batch_size: Batch size
            output_dir: Output directory
        """
        optimizer_policy = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.learning_rate
        )
        optimizer_value = torch.optim.AdamW(
            list(self.value_model.parameters()) + list(self.value_head.parameters()),
            lr=self.learning_rate
        )

        logger.info("Starting RLHF training...")

        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")

            # Generate responses
            all_prompts = prompts * num_samples_per_prompt
            responses = self.generate_responses(prompts, num_samples_per_prompt)

            # Compute rewards
            rewards = self.compute_rewards(all_prompts, responses)

            # Prepare data for PPO
            states, actions, old_log_probs = self._prepare_ppo_data(all_prompts, responses)

            # Compute values and advantages
            with torch.no_grad():
                values = self._compute_values(states)

            advantages = self.compute_advantages(rewards, values)
            returns = advantages + values

            # PPO optimization
            dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch

                # Update policy
                optimizer_policy.zero_grad()
                policy_loss = self.ppo_step(
                    batch_old_log_probs, batch_states, batch_actions,
                    batch_advantages, batch_returns
                )
                policy_loss.backward()
                optimizer_policy.step()

                # Update value function
                optimizer_value.zero_grad()
                value_loss = self._compute_value_loss(batch_states, batch_returns)
                value_loss.backward()
                optimizer_value.step()

            logger.info(f"Iteration {iteration + 1} completed")

        # Save final model
        self.save_model(output_dir)
        logger.info("RLHF training completed!")

    def _prepare_ppo_data(self, prompts: List[str], responses: List[str]):
        """Prepare data for PPO training."""
        states = []
        actions = []
        old_log_probs = []

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}\n\n{response}"
            tokens = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.policy_model(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens['attention_mask']
                )
                logits = outputs.logits

            # Get log probs for response tokens
            prompt_length = len(self.tokenizer(prompt)['input_ids'])
            response_logits = logits[0, prompt_length-1:-1]
            response_tokens = tokens['input_ids'][0, prompt_length:]

            log_probs = torch.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            states.append(tokens['input_ids'][0])
            actions.append(response_tokens)
            old_log_probs.append(token_log_probs)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(old_log_probs)
        )

    def _compute_values(self, states: torch.Tensor) -> torch.Tensor:
        """Compute value estimates."""
        with torch.no_grad():
            outputs = self.value_model(
                input_ids=states,
                attention_mask=(states != self.tokenizer.pad_token_id).long(),
                output_hidden_states=True
            )
            values = self.value_head(outputs.hidden_states[-1][:, -1, :])
        return values.squeeze()

    def _compute_value_loss(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        outputs = self.value_model(
            input_ids=states,
            attention_mask=(states != self.tokenizer.pad_token_id).long(),
            output_hidden_states=True
        )
        values = self.value_head(outputs.hidden_states[-1][:, -1, :])
        return nn.MSELoss()(values.squeeze(), returns)

    def save_model(self, output_dir: str):
        """Save the trained model."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.policy_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model to {output_dir}")


def create_sample_prompts() -> List[str]:
    """Create sample prompts for RLHF training."""
    return [
        "What is the capital of France?",
        "Explain Newton's first law of motion.",
        "What is the speed of light?",
        "Why is the sky blue?",
        "What is photosynthesis?"
    ]


if __name__ == "__main__":
    # Example usage (requires a trained reward model)
    prompts = create_sample_prompts()

    # Note: In practice, you would load a trained reward model
    # reward_model = RewardModel("gpt2")
    # trainer = RLHFTrainer("gpt2", reward_model)

    print("RLHF trainer created. Note: Requires trained reward model for actual training.")
