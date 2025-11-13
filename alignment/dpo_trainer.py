"""Direct Preference Optimization (DPO) trainer for aligning language models."""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for preference learning with chosen and rejected responses."""

    def __init__(
        self,
        tokenizer,
        data: List[Dict[str, str]],
        max_length: int = 512
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            data: List of dicts with keys: 'prompt', 'chosen', 'rejected'
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt + chosen response
        prompt_chosen = f"{item['prompt']}\n\n{item['chosen']}"
        chosen_tokens = self.tokenizer(
            prompt_chosen,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize prompt + rejected response
        prompt_rejected = f"{item['prompt']}\n\n{item['rejected']}"
        rejected_tokens = self.tokenizer(
            prompt_rejected,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Get prompt length for masking
        prompt_tokens = self.tokenizer(
            item['prompt'],
            return_tensors="pt"
        )
        prompt_length = len(prompt_tokens['input_ids'][0])

        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
            'prompt_length': prompt_length
        }


class DPOTrainer:
    """Direct Preference Optimization trainer."""

    def __init__(
        self,
        model_name: str,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model name or path
            beta: DPO regularization parameter
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.beta = beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_length = max_length

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ref_model.to(self.device)

        logger.info(f"Initialized DPO trainer with model: {model_name}")
        logger.info(f"Using device: {self.device}")

    def compute_dpo_loss(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """
        Compute DPO loss.

        Args:
            chosen_logits: Logits for chosen responses
            rejected_logits: Logits for rejected responses
            chosen_attention_mask: Attention mask for chosen responses
            rejected_attention_mask: Attention mask for rejected responses
            prompt_length: Length of prompt tokens

        Returns:
            DPO loss
        """
        # Get log probabilities for chosen and rejected responses
        chosen_log_probs = self._get_log_probs(
            chosen_logits, chosen_attention_mask, prompt_length
        )
        rejected_log_probs = self._get_log_probs(
            rejected_logits, rejected_attention_mask, prompt_length
        )

        # Get reference model log probabilities
        with torch.no_grad():
            ref_chosen_log_probs = self._get_ref_log_probs(
                chosen_logits, chosen_attention_mask, prompt_length
            )
            ref_rejected_log_probs = self._get_ref_log_probs(
                rejected_logits, rejected_attention_mask, prompt_length
            )

        # Compute DPO loss
        chosen_diff = chosen_log_probs - ref_chosen_log_probs
        rejected_diff = rejected_log_probs - ref_rejected_log_probs

        loss = -torch.log(torch.sigmoid(self.beta * (chosen_diff - rejected_diff)))
        return loss.mean()

    def _get_log_probs(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Get log probabilities for the response part only."""
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = logits[..., 1:, :].contiguous()

        # Get log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)

        # Only consider response tokens (after prompt)
        response_mask = attention_mask[..., prompt_length:-1]
        response_log_probs = log_probs[..., prompt_length-1:-1, :]

        # Gather log probs for the actual labels
        labels = shift_labels[..., prompt_length-1:, :]
        gathered_log_probs = torch.gather(
            response_log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply attention mask
        masked_log_probs = gathered_log_probs * response_mask

        return masked_log_probs.sum(dim=-1) / response_mask.sum(dim=-1)

    def _get_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Get log probabilities from reference model."""
        with torch.no_grad():
            outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return self._get_log_probs(
                outputs.logits, attention_mask, prompt_length
            )

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
        batch_size: int = 4,
        num_epochs: int = 3,
        save_steps: int = 100,
        output_dir: str = "./dpo_results"
    ):
        """
        Train the model using DPO.

        Args:
            train_data: Training preference data
            val_data: Validation preference data
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            save_steps: Save model every N steps
            output_dir: Directory to save model checkpoints
        """
        # Create datasets
        train_dataset = PreferenceDataset(self.tokenizer, train_data, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data:
            val_dataset = PreferenceDataset(self.tokenizer, val_data, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        self.model.train()
        global_step = 0

        logger.info("Starting DPO training...")

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in train_loader:
                # Move batch to device
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                prompt_length = batch['prompt_length'].to(self.device)

                # Get model outputs
                chosen_outputs = self.model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                rejected_outputs = self.model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )

                # Compute DPO loss
                loss = self.compute_dpo_loss(
                    chosen_outputs.logits, rejected_outputs.logits,
                    chosen_attention_mask, rejected_attention_mask,
                    prompt_length
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % save_steps == 0:
                    self.save_checkpoint(output_dir, global_step)

            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

            # Validation
            if val_data:
                val_loss = self.validate(val_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}")

        # Save final model
        self.save_checkpoint(output_dir, global_step, final=True)
        logger.info("DPO training completed!")

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                prompt_length = batch['prompt_length'].to(self.device)

                chosen_outputs = self.model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                rejected_outputs = self.model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )

                loss = self.compute_dpo_loss(
                    chosen_outputs.logits, rejected_outputs.logits,
                    chosen_attention_mask, rejected_attention_mask,
                    prompt_length
                )

                total_loss += loss.item()

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
    trainer = DPOTrainer("gpt2")  # Use a smaller model for testing
    sample_data = create_sample_preference_data()

    trainer.train(
        train_data=sample_data,
        batch_size=2,
        num_epochs=1,
        output_dir="./dpo_test_results"
    )
