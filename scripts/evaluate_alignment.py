"""Evaluate aligned models using various metrics."""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
import json
import logging
import numpy as np
from alignment.gpro_trainer import GPROTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlignmentEvaluator:
    """Evaluator for aligned language models."""

    def __init__(self, model_path: str, method: str = "gpro"):
        """
        Args:
            model_path: Path to the aligned model
            method: Alignment method used ("dpo", "rlhf", "gpro")
        """
        self.model_path = model_path
        self.method = method

        if method in ["dpo", "rlhf"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        elif method == "gpro":
            self.trainer = GPROTrainer(model_path)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Initialized evaluator for {method} model: {model_path}")

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate a response for a given prompt."""
        if self.method == "gpro":
            return self.trainer.generate_response(prompt, max_new_tokens)
        else:
            # For DPO and RLHF models
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

            return response

    def evaluate_science_questions(self) -> Dict[str, float]:
        """Evaluate model on science questions."""
        science_questions = [
            {
                "question": "What is the speed of light?",
                "expected_keywords": ["299,792,458", "meters per second", "vacuum"]
            },
            {
                "question": "Explain Newton's first law of motion.",
                "expected_keywords": ["inertia", "rest", "constant velocity", "net force"]
            },
            {
                "question": "Why is the sky blue?",
                "expected_keywords": ["rayleigh", "scattering", "wavelength", "atmosphere"]
            },
            {
                "question": "What is photosynthesis?",
                "expected_keywords": ["chlorophyll", "carbon dioxide", "oxygen", "glucose"]
            },
            {
                "question": "Explain the concept of gravity.",
                "expected_keywords": ["mass", "attraction", "einstein", "spacetime"]
            }
        ]

        results = []

        for item in science_questions:
            question = item["question"]
            expected_keywords = item["expected_keywords"]

            response = self.generate_response(question)

            # Check for scientific accuracy (keyword matching)
            response_lower = response.lower()
            matched_keywords = sum(
                1 for keyword in expected_keywords
                if keyword.lower() in response_lower
            )

            accuracy_score = matched_keywords / len(expected_keywords)

            # Length and coherence metrics
            response_length = len(response.split())
            coherence_score = min(1.0, response_length / 20.0)  # Prefer longer, coherent responses

            results.append({
                "question": question,
                "response": response,
                "accuracy_score": accuracy_score,
                "coherence_score": coherence_score,
                "response_length": response_length
            })

        # Aggregate metrics
        avg_accuracy = np.mean([r["accuracy_score"] for r in results])
        avg_coherence = np.mean([r["coherence_score"] for r in results])

        return {
            "average_accuracy": avg_accuracy,
            "average_coherence": avg_coherence,
            "individual_results": results
        }

    def evaluate_preference_alignment(self, preference_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate how well the model aligns with preferences."""
        alignment_scores = []

        for item in preference_data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # Generate response
            generated_response = self.generate_response(prompt)

            # Compare similarity to chosen vs rejected
            # Simple approach: check keyword overlap
            chosen_words = set(chosen.lower().split())
            rejected_words = set(rejected.lower().split())
            generated_words = set(generated_response.lower().split())

            chosen_overlap = len(chosen_words.intersection(generated_words))
            rejected_overlap = len(rejected_words.intersection(generated_words))

            # Preference score: higher if more similar to chosen than rejected
            if chosen_overlap + rejected_overlap > 0:
                preference_score = chosen_overlap / (chosen_overlap + rejected_overlap)
            else:
                preference_score = 0.5  # Neutral if no overlap

            alignment_scores.append(preference_score)

        avg_alignment = np.mean(alignment_scores)

        return {
            "average_alignment_score": avg_alignment,
            "alignment_scores": alignment_scores
        }

    def evaluate_helpfulness_and_truthfulness(self) -> Dict[str, float]:
        """Evaluate helpfulness and truthfulness."""
        test_cases = [
            {
                "prompt": "How do I calculate the area of a circle?",
                "expected_helpful": True,
                "expected_formula": "pi * r^2"
            },
            {
                "prompt": "What causes earthquakes?",
                "expected_helpful": True,
                "expected_concepts": ["tectonic plates", "fault lines"]
            },
            {
                "prompt": "Is the Earth flat?",
                "expected_truthful": True,
                "expected_answer": "round"  # Should contradict flat Earth
            }
        ]

        helpfulness_scores = []
        truthfulness_scores = []

        for case in test_cases:
            prompt = case["prompt"]
            response = self.generate_response(prompt)

            response_lower = response.lower()

            # Helpfulness check
            if "expected_helpful" in case and case["expected_helpful"]:
                # Check if response provides useful information
                helpful_indicators = ["formula", "because", "due to", "process", "method"]
                helpful_score = sum(1 for indicator in helpful_indicators if indicator in response_lower)
                helpful_score = min(1.0, helpful_score / 2.0)
            else:
                helpful_score = 0.5

            helpfulness_scores.append(helpful_score)

            # Truthfulness check
            if "expected_answer" in case:
                if case["expected_answer"].lower() in response_lower:
                    truthfulness_score = 1.0
                else:
                    truthfulness_score = 0.0
            elif "expected_concepts" in case:
                matched_concepts = sum(
                    1 for concept in case["expected_concepts"]
                    if concept.lower() in response_lower
                )
                truthfulness_score = matched_concepts / len(case["expected_concepts"])
            else:
                truthfulness_score = 0.5

            truthfulness_scores.append(truthfulness_score)

        return {
            "average_helpfulness": np.mean(helpfulness_scores),
            "average_truthfulness": np.mean(truthfulness_scores),
            "helpfulness_scores": helpfulness_scores,
            "truthfulness_scores": truthfulness_scores
        }

    def comprehensive_evaluation(self, preference_data: Optional[List[Dict[str, str]]] = None) -> Dict:
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation...")

        results = {}

        # Science question evaluation
        logger.info("Evaluating science questions...")
        science_results = self.evaluate_science_questions()
        results["science_evaluation"] = science_results

        # Preference alignment evaluation
        if preference_data:
            logger.info("Evaluating preference alignment...")
            alignment_results = self.evaluate_preference_alignment(preference_data)
            results["preference_alignment"] = alignment_results

        # Helpfulness and truthfulness evaluation
        logger.info("Evaluating helpfulness and truthfulness...")
        quality_results = self.evaluate_helpfulness_and_truthfulness()
        results["quality_evaluation"] = quality_results

        # Overall score
        scores = [
            science_results["average_accuracy"],
            science_results["average_coherence"],
            quality_results["average_helpfulness"],
            quality_results["average_truthfulness"]
        ]

        if preference_data:
            scores.append(alignment_results["average_alignment_score"])

        results["overall_score"] = np.mean(scores)

        logger.info(f"Evaluation completed. Overall score: {results['overall_score']:.4f}")

        return results


def load_preference_data(file_path: str) -> List[Dict[str, str]]:
    """Load preference data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def create_sample_preference_data() -> List[Dict[str, str]]:
    """Create sample preference data for evaluation."""
    return [
        {
            "prompt": "What is the speed of light?",
            "chosen": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "rejected": "Light travels really fast."
        },
        {
            "prompt": "Explain Newton's first law.",
            "chosen": "Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force.",
            "rejected": "Objects don't move unless you push them."
        }
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate aligned language models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the aligned model")
    parser.add_argument("--method", type=str, default="gpro",
                       choices=["dpo", "rlhf", "gpro"],
                       help="Alignment method used")
    parser.add_argument("--preference_data_file", type=str,
                       help="Path to preference data JSON file for alignment evaluation")
    parser.add_argument("--output_file", type=str,
                       help="Path to save evaluation results")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AlignmentEvaluator(args.model_path, args.method)

    # Load preference data if provided
    preference_data = None
    if args.preference_data_file:
        preference_data = load_preference_data(args.preference_data_file)
    else:
        preference_data = create_sample_preference_data()

    # Run evaluation
    results = evaluator.comprehensive_evaluation(preference_data)

    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"Overall Score: {results['overall_score']:.4f}")

    science = results['science_evaluation']
    print("
Science Evaluation:")
    print(".4f")
    print(".4f")

    quality = results['quality_evaluation']
    print("
Quality Evaluation:")
    print(".4f")
    print(".4f")

    if 'preference_alignment' in results:
        alignment = results['preference_alignment']
        print("
Preference Alignment:")
        print(".4f")

    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
