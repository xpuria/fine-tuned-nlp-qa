"""Unified alignment training script for DPO, RLHF, and GPRO."""

import argparse
import json
import logging
from typing import Dict, List, Optional
import torch

from alignment.dpo_trainer import DPOTrainer
from alignment.rlhf_trainer import RLHFTrainer, RewardModel
from alignment.gpro_trainer import GPROTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_preference_data(file_path: str) -> List[Dict[str, str]]:
    """Load preference data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def create_sample_science_preference_data() -> List[Dict[str, str]]:
    """Create sample preference data focused on science questions."""
    return [
        {
            "prompt": "What is the speed of light?",
            "chosen": "The speed of light in vacuum is approximately 299,792,458 meters per second, which is a fundamental constant denoted as 'c' in physics.",
            "rejected": "Light travels really fast, like 300 million meters per second or something."
        },
        {
            "prompt": "Explain Newton's first law of motion.",
            "chosen": "Newton's first law of motion, also known as the law of inertia, states that an object at rest will remain at rest, and an object in motion will remain in motion at constant velocity, unless acted upon by an unbalanced net force.",
            "rejected": "Newton's first law says that objects don't move unless you push them."
        },
        {
            "prompt": "Why is the sky blue?",
            "chosen": "The sky appears blue due to Rayleigh scattering, where shorter wavelength blue light is scattered more by the atmosphere than longer wavelength red light, making the sky look blue during the day.",
            "rejected": "The sky is blue because it reflects the ocean."
        },
        {
            "prompt": "What is photosynthesis?",
            "chosen": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose, using carbon dioxide and water, while releasing oxygen as a byproduct.",
            "rejected": "Photosynthesis is when plants make food from sunlight."
        },
        {
            "prompt": "Explain the concept of gravity.",
            "chosen": "Gravity is a fundamental force of nature that causes mutual attraction between all objects with mass. According to Einstein's theory of general relativity, gravity is the curvature of spacetime caused by mass and energy.",
            "rejected": "Gravity is what makes things fall down to Earth."
        },
        {
            "prompt": "What is the periodic table?",
            "chosen": "The periodic table is a tabular arrangement of chemical elements, organized by atomic number, electron configuration, and recurring chemical properties. Elements are arranged in rows (periods) and columns (groups) based on their electron structure.",
            "rejected": "The periodic table is a chart with all the chemical elements listed."
        },
        {
            "prompt": "How does electricity work?",
            "chosen": "Electricity involves the flow of electric charge through conductors. It can be generated through various means like electromagnetic induction, and is transmitted through power grids to provide energy for lighting, heating, and powering devices.",
            "rejected": "Electricity is like magic that makes lights turn on."
        },
        {
            "prompt": "What is evolution?",
            "chosen": "Evolution is the process by which species of organisms change over time through genetic variation and natural selection. It explains the diversity of life on Earth and how species adapt to their environments over generations.",
            "rejected": "Evolution is when animals change into other animals over time."
        }
    ]


def train_dpo(args):
    """Train using Direct Preference Optimization."""
    logger.info("Starting DPO training...")

    # Load data
    if args.data_file:
        train_data = load_preference_data(args.data_file)
    else:
        train_data = create_sample_science_preference_data()

    # Split data for validation
    val_data = train_data[-2:] if len(train_data) > 4 else None
    train_data = train_data[:-2] if len(train_data) > 4 else train_data

    # Initialize trainer
    trainer = DPOTrainer(
        model_name=args.model_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length
    )

    # Train
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        output_dir=args.output_dir
    )


def train_rlhf(args):
    """Train using Reinforcement Learning from Human Feedback."""
    logger.info("Starting RLHF training...")

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
    else:
        prompts = [item['prompt'] for item in create_sample_science_preference_data()]

    # Load or initialize reward model
    if args.reward_model_path:
        reward_model = RewardModel(args.model_name)
        reward_model.load_state_dict(torch.load(args.reward_model_path))
    else:
        logger.warning("No reward model provided. Using untrained reward model.")
        reward_model = RewardModel(args.model_name)

    # Initialize trainer
    trainer = RLHFTrainer(
        model_name=args.model_name,
        reward_model=reward_model,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

    # Train
    trainer.train(
        prompts=prompts,
        num_iterations=args.num_iterations,
        num_samples_per_prompt=args.num_samples_per_prompt,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )


def train_gpro(args):
    """Train using Gradient-based Preference Optimization."""
    logger.info("Starting GPRO training...")

    # Load data
    if args.data_file:
        train_data = load_preference_data(args.data_file)
    else:
        train_data = create_sample_science_preference_data()

    # Split data for validation
    val_data = train_data[-2:] if len(train_data) > 4 else None
    train_data = train_data[:-2] if len(train_data) > 4 else train_data

    # Initialize trainer
    trainer = GPROTrainer(
        model_name=args.model_name,
        tau=args.tau,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length
    )

    # Train
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_penalty_coef=args.gradient_penalty_coef,
        save_steps=args.save_steps,
        output_dir=args.output_dir
    )


def main():
    parser = argparse.ArgumentParser(description="Train language models with alignment methods")

    # Common arguments
    parser.add_argument("--method", type=str, required=True,
                       choices=["dpo", "rlhf", "gpro"],
                       help="Alignment method to use")
    parser.add_argument("--model_name", type=str, default="t5-base",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./alignment_results",
                       help="Output directory")
    parser.add_argument("--data_file", type=str,
                       help="Path to preference data JSON file")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save model every N steps")

    # DPO and GPRO specific arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")

    # DPO specific arguments
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO regularization parameter")

    # RLHF specific arguments
    parser.add_argument("--prompts_file", type=str,
                       help="Path to prompts JSON file")
    parser.add_argument("--reward_model_path", type=str,
                       help="Path to trained reward model")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="Number of RLHF iterations")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4,
                       help="Number of samples per prompt")
    parser.add_argument("--kl_coef", type=float, default=0.2,
                       help="KL divergence coefficient")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="PPO clipping range")
    parser.add_argument("--value_coef", type=float, default=0.5,
                       help="Value function coefficient")

    # GPRO specific arguments
    parser.add_argument("--tau", type=float, default=1.0,
                       help="Temperature parameter for GPRO")
    parser.add_argument("--gradient_penalty_coef", type=float, default=0.1,
                       help="Gradient penalty coefficient")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Train based on method
    if args.method == "dpo":
        train_dpo(args)
    elif args.method == "rlhf":
        train_rlhf(args)
    elif args.method == "gpro":
        train_gpro(args)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
