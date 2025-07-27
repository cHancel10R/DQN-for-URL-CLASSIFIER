
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Dict
import numpy as np

from preprocessor import URLPreprocessor
from environment import URLEnvironment
from dqn_model import DURLD_DQN

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Experiment management for DRL-based URL detection"""
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    
    def load_data(self):
        """Load and split dataset with label preprocessing"""
        # Read CSV, assuming no header
        data = pd.read_csv(self.data_path, header=None, names=['url', 'label'])
        assert 'url' in data.columns and 'label' in data.columns, "Missing required columns"
        
        # Map labels: benign -> 0, others (defacement, malware, phishing) -> 1
        data['label'] = data['label'].apply(lambda x: 0 if x == 'benign' else 1)
        
        urls = data['url'].values
        labels = data['label'].values
        
        self.preprocessor = URLPreprocessor()
        self.preprocessor.create_char_dict(urls)
        X = self.preprocessor.preprocess_urls(urls)
        y = labels
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Data loaded: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def run_experiment(self, model_type: str = 'dqn', episodes: int = 100, batch_size: int = 64, **kwargs) -> Dict:
        if not all([self.X_train, self.X_test, self.y_train, self.y_test]):
            self.load_data()
        
        if model_type == 'dqn':
            model = DURLD_DQN(
                vocab_size=self.preprocessor.vocab_size,
                max_len=self.preprocessor.max_len,
                embedding_dim=kwargs.get('embedding_dim', 128),
                model_name=kwargs.get('model_name', 'durld_dqn')
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        env = URLEnvironment(self.X_train, self.y_train)
        history = model.train(env, episodes=episodes, batch_size=batch_size)
        metrics = model.evaluate(self.X_test, self.y_test)
        
        model.plot_training_history()
        model.plot_confusion_matrix(metrics['confusion_matrix'])
        model.plot_roc_pr_curves(self.X_test, self.y_test)
        
        logger.info(f"Experiment parameters: {kwargs}")
        logger.info(f"Evaluation metrics: {metrics}")
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'preprocessor': self.preprocessor
        }