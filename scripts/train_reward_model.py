"""Train reward model for RLHF."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    TrainingArguments,
    Trainer
)
from typing import Dict, List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardDataset(Dataset):
    """Dataset for reward model training."""

    def __init__(
        self,
        tokenizer,
        data: List[Dict[str, str]],
        max_length: int = 512
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            data: List of dicts with keys: 'chosen', 'rejected'
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize chosen response
        chosen_tokens = self.tokenizer(
            item["chosen"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize rejected response
        rejected_tokens = self.tokenizer(
            item["rejected"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }


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


def compute_reward_loss(
    reward_model: RewardModel,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute reward model loss using preference pairs."""
    # Get rewards for chosen and rejected responses
    chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
    rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)

    # Loss: maximize reward for chosen, minimize for rejected
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss


def train_reward_model(
    model_name: str,
    train_data: List[Dict[str, str]],
    val_data: Optional[List[Dict[str, str]]] = None,
    output_dir: str = "./reward_model",
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    weight_decay: float = 0.01
):
    """
    Train the reward model.

    Args:
        model_name: Base model name
        train_data: Training preference data
        val_data: Validation preference data
        output_dir: Output directory
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
        max_length: Maximum sequence length
        weight_decay: Weight decay
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_model = RewardModel(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    # Create datasets
    train_dataset = RewardDataset(tokenizer, train_data, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data:
        val_dataset = RewardDataset(tokenizer, val_data, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    reward_model.train()
    logger.info("Starting reward model training...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in train_loader:
            # Move batch to device
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            # Compute loss
            loss = compute_reward_loss(
                reward_model,
                chosen_input_ids, chosen_attention_mask,
                rejected_input_ids, rejected_attention_mask
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

        # Validation
        if val_data:
            val_loss = validate_reward_model(reward_model, val_loader, device)
            logger.info(f"Validation Loss: {val_loss:.4f}")

    # Save model
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save the reward model weights
    torch.save(reward_model.state_dict(), os.path.join(output_dir, "reward_model.pt"))
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Reward model saved to {output_dir}")


def validate_reward_model(
    reward_model: RewardModel,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """Validate the reward model."""
    reward_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            loss = compute_reward_loss(
                reward_model,
                chosen_input_ids, chosen_attention_mask,
                rejected_input_ids, rejected_attention_mask
            )

            total_loss += loss.item()

    reward_model.train()
    return total_loss / len(val_loader)


def create_sample_reward_data() -> List[Dict[str, str]]:
    """Create sample preference data for reward model training."""
    return [
        {
            "chosen": "The speed of light in vacuum is exactly 299,792,458 meters per second, as defined by the International System of Units.",
            "rejected": "Light goes really fast, like 300 million meters per second."
        },
        {
            "chosen": "Newton's first law states that an object at rest remains at rest, and an object in motion continues in motion with constant velocity unless acted upon by a net external force.",
            "rejected": "Stuff doesn't move unless you push it."
        },
        {
            "chosen": "The sky appears blue because of Rayleigh scattering, where molecules in the atmosphere scatter shorter-wavelength blue light more effectively than longer-wavelength red light.",
            "rejected": "The sky is blue because the ocean is blue and it reflects up."
        },
        {
            "chosen": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water, producing oxygen as a byproduct.",
            "rejected": "Plants make food from sunlight and air."
        }
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train reward model for RLHF")
    parser.add_argument("--model_name", type=str, default="gpt2",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./reward_model",
                       help="Output directory")
    parser.add_argument("--data_file", type=str,
                       help="Path to preference data JSON file")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")

    args = parser.parse_args()

    # Load data
    if args.data_file:
        with open(args.data_file, 'r') as f:
            train_data = json.load(f)
    else:
        train_data = create_sample_reward_data()

    # Split for validation
    val_data = train_data[-1:] if len(train_data) > 3 else None
    train_data = train_data[:-1] if len(train_data) > 3 else train_data

    # Train reward model
    train_reward_model(
        model_name=args.model_name,
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        weight_decay=args.weight_decay
    )


if __name__ == "__main__":
    main()
