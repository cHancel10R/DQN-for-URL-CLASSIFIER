import logging
import pandas as pd
import os
from experiment import ExperimentRunner

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler('experiment_drl.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

def run_tests():
    logger = logging.getLogger(__name__)
    logger.info("Running test suite")
    sample_data = pd.DataFrame({
        'url': [
            'http://example.com',
            'http://malicious.com/phish',
            'https://safe.org'
        ],
        'label': [0, 1, 0]
    })
    sample_data.to_csv('test_data.csv', index=False)
    
    experiment = ExperimentRunner('test_data.csv')
    experiment.load_data()
    results = experiment.run_experiment(model_type='dqn', episodes=2)
    
    assert results['metrics']['accuracy'] > 0, "Accuracy should be positive"
    assert os.path.exists('dqn_training_history.png'), "Training history plot missing"
    logger.info("Test suite passed")