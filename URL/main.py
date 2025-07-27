import sys

print("Python executable:", sys.executable)
print("sys.path:", sys.path)

import argparse
import logging
from utils import setup_logging, run_tests
from experiment import ExperimentRunner

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="DeepURLDetect DRL Experiment Runner")
    parser.add_argument(
        "--data_path", default="data/malicious_urls_dataset.csv", help="Path to dataset"
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--run_tests", action="store_true", help="Run test suite")
    args = parser.parse_args()

    try:
        if args.run_tests:
            run_tests()
            return

        experiment = ExperimentRunner(
            data_path=args.data_path, test_size=args.test_size
        )
        results = experiment.run_experiment(
            model_type="dqn",
            episodes=args.episodes,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            model_name="durld_dqn",
        )

        results["model"].save()
        results["preprocessor"].save("preprocessor.pkl")

        test_url = "http://malicious.com/phishing/page.html"
        proba, label = results["model"].predict_url(test_url, results["preprocessor"])
        logger.info(f"URL: {test_url}")
        logger.info(f"Prediction: {label} (Probability: {proba:.4f})")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
