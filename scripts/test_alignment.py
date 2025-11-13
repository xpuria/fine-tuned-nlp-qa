"""Quick test script for alignment methods."""

import sys
import os

# Add the alignment module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from alignment.dpo_trainer import create_sample_preference_data as create_dpo_data
from alignment.gpro_trainer import create_sample_preference_data as create_gpro_data


def test_imports():
    """Test that all alignment modules can be imported."""
    try:
        from alignment.dpo_trainer import DPOTrainer
        from alignment.rlhf_trainer import RLHFTrainer
        from alignment.gpro_trainer import GPROTrainer
        print("✓ All alignment modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_creation():
    """Test that preference data can be created."""
    try:
        dpo_data = create_dpo_data()
        gpro_data = create_gpro_data()

        assert len(dpo_data) > 0, "DPO data is empty"
        assert len(gpro_data) > 0, "GPRO data is empty"

        # Check data structure
        for item in dpo_data:
            assert "prompt" in item, "Missing prompt in DPO data"
            assert "chosen" in item, "Missing chosen in DPO data"
            assert "rejected" in item, "Missing rejected in DPO data"

        print("✓ Preference data creation successful")
        return True
    except Exception as e:
        print(f"✗ Data creation error: {e}")
        return False


def test_basic_training():
    """Test basic training initialization (without actually training)."""
    try:
        from alignment.dpo_trainer import DPOTrainer
        from alignment.gpro_trainer import GPROTrainer

        # Test DPO trainer initialization
        dpo_trainer = DPOTrainer("gpt2", beta=0.1, max_length=128)

        # Test GPRO trainer initialization
        gpro_trainer = GPROTrainer("gpt2", tau=1.0, max_length=128)

        print("✓ Trainer initialization successful")
        return True
    except Exception as e:
        print(f"✗ Trainer initialization error: {e}")
        return False


def main():
    """Run all tests."""
    print("Running alignment method tests...\n")

    tests = [
        test_imports,
        test_data_creation,
        test_basic_training
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Alignment methods are ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
