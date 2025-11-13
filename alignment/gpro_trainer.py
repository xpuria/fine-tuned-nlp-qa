"""Gradient-based Preference Optimization (GPRO) trainer."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPROTrainer:
    """Gradient-based Preference Optimization trainer."""

    def __init__(
        self,
        model_name: str,
        tau: float = 1.0,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace model name or path
            tau: Temperature parameter for preference modeling
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.tau = tau
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_length = max_length

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info("Initialized GPRO trainer with model: %s", model_name)
        logger.info("Using device: %s", self.device)

    def compute_preference_probabilities(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference probability for chosen responses.

        Returns:
            Probability that the chosen response is preferred over the rejected one.
        """
        chosen_log_probs = self._get_sequence_log_probs(
            chosen_logits, chosen_attention_mask
        )
        rejected_log_probs = self._get_sequence_log_probs(
            rejected_logits, rejected_attention_mask
        )

        # Temperature scaling and logistic preference
        score_diff = (chosen_log_probs - rejected_log_probs) / self.tau
        chosen_probs = torch.sigmoid(score_diff)
        return chosen_probs

    @staticmethod
    def _get_sequence_log_probs(
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probabilities for entire sequences."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = logits[..., 1:, :].contiguous()

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Exclude padding tokens
        token_mask = attention_mask[..., 1:]
        masked_log_probs = gathered_log_probs * token_mask

        sequence_log_probs = masked_log_probs.sum(dim=-1)
        sequence_lengths = token_mask.sum(dim=-1).clamp(min=1)
        return sequence_log_probs / sequence_lengths

    def compute_gpro_loss(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the primary GPRO loss."""
        chosen_probs = self.compute_preference_probabilities(
            chosen_logits,
            rejected_logits,
            chosen_attention_mask,
            rejected_attention_mask,
        )
        # Encourage probability mass on the preferred sample
        loss = -torch.log(chosen_probs.clamp(min=1e-8)).mean()
        return loss

    def compute_gradient_penalty(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient penalty to encourage smooth preference surfaces.
        """
        chosen_probs = self.compute_preference_probabilities(
            chosen_logits,
            rejected_logits,
            chosen_attention_mask,
            rejected_attention_mask,
        )

        grad_outputs = torch.ones_like(chosen_probs, device=chosen_probs.device)
        gradients = torch.autograd.grad(
            outputs=chosen_probs,
            inputs=(chosen_logits, rejected_logits),
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        chosen_grad, rejected_grad = gradients
        if chosen_grad is None or rejected_grad is None:
            return torch.tensor(0.0, device=chosen_probs.device)

        penalty = (
            chosen_grad.pow(2).sum(dim=list(range(1, chosen_grad.ndim)))
            + rejected_grad.pow(2).sum(dim=list(range(1, rejected_grad.ndim)))
        )
        return penalty.mean()

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
        batch_size: int = 4,
        num_epochs: int = 3,
        gradient_penalty_coef: float = 0.1,
        save_steps: int = 100,
        output_dir: str = "./gpro_results",
    ):
        """Train the model using GPRO."""
        from alignment.dpo_trainer import PreferenceDataset

        train_dataset = PreferenceDataset(self.tokenizer, train_data, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data:
            val_dataset = PreferenceDataset(self.tokenizer, val_data, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        logger.info("Starting GPRO training...")
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)

                chosen_outputs = self.model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                )
                rejected_outputs = self.model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                )

                gpro_loss = self.compute_gpro_loss(
                    chosen_outputs.logits,
                    rejected_outputs.logits,
                    chosen_attention_mask,
                    rejected_attention_mask,
                )

                grad_penalty = self.compute_gradient_penalty(
                    chosen_outputs.logits,
                    rejected_outputs.logits,
                    chosen_attention_mask,
                    rejected_attention_mask,
                )

                total_loss = gpro_loss + gradient_penalty_coef * grad_penalty
                total_loss.backward()

                optimizer.step()
                scheduler.step()

                epoch_loss += total_loss.item()
                global_step += 1

                if global_step % save_steps == 0:
                    self.save_checkpoint(output_dir, global_step)

            avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
            logger.info("Epoch %s/%s, Loss: %.4f", epoch + 1, num_epochs, avg_epoch_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                logger.info("Validation Loss: %.4f", val_loss)

        self.save_checkpoint(output_dir, global_step, final=True)
        logger.info("GPRO training completed!")

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model (without gradient penalty)."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)

                chosen_outputs = self.model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                rejected_outputs = self.model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )

                gpro_loss = self.compute_gpro_loss(
                    chosen_outputs.logits, rejected_outputs.logits,
                    chosen_attention_mask, rejected_attention_mask
                )

                total_loss += gpro_loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def save_checkpoint(self, output_dir: str, step: int, final: bool = False):
        """Save model checkpoint."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_name = "final" if final else f"checkpoint-{step}"
        checkpoint_path = os.path.join(output_dir, checkpoint_name)

        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate a response for a given prompt."""
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )

        self.model.train()
        return response


def create_sample_preference_data() -> List[Dict[str, str]]:
    """Create sample preference data for testing."""
    return [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "I think it's Paris, but I'm not sure."
        },
        {
            "prompt": "Explain Newton's first law.",
            "chosen": "Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
            "rejected": "It's about objects not moving unless pushed."
        },
        {
            "prompt": "What is the speed of light?",
            "chosen": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "rejected": "Light travels at 300,000,000 meters per second."
        }
    ]


if __name__ == "__main__":
    # Example usage
    trainer = GPROTrainer("gpt2")  # Use a smaller model for testing
    sample_data = create_sample_preference_data()

    trainer.train(
        train_data=sample_data,
        batch_size=2,
        num_epochs=1,
        output_dir="./gpro_test_results"
    )

    # Test generation
    test_prompt = "What is the capital of France?"
    response = trainer.generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
