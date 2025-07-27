
import numpy as np
from typing import Tuple

class URLEnvironment:
    def __init__(self, X: np.ndarray, y: np.ndarray, episode_samples: int = 1000):
        self.X = X
        self.y = y
        self.episode_samples = min(episode_samples, len(X))
        self.n_samples = len(X)
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.current_idx = 0
        indices = np.random.choice(self.n_samples, self.episode_samples, replace=False)
        self.current_X = self.X[indices]
        self.current_y = self.y[indices]
        return self.current_X[0]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        true_label = self.current_y[self.current_idx]
        reward = 2.0 if action == true_label and true_label == 1 else (1.0 if action == true_label else -1.0)
        self.current_idx += 1
        done = self.current_idx >= len(self.current_X)
        next_state = self.current_X[self.current_idx] if not done else np.zeros_like(self.current_X[0])
        return next_state, reward, done