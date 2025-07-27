import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from preprocessor import URLPreprocessor

logger = logging.getLogger(__name__)


class DURLD_DQN:
    """Deep Q-Network for URL detection"""

    def __init__(
        self,
        vocab_size: int = 150,
        max_len: int = 500,
        embedding_dim: int = 128,
        model_name: str = "durld_dqn",
        replay_buffer_size: int = 10000,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.history = None

    def _build_model(self) -> Model:
        inputs = Input(shape=(self.max_len,), name="input_layer")
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            mask_zero=True,
            name="char_embedding",
        )(inputs)
        x = SpatialDropout1D(0.2)(x)

        conv1 = Conv1D(64, 3, activation="relu", padding="same", name="conv1d_3gram")(x)
        conv1 = MaxPooling1D(pool_size=2, name="maxpool_3gram")(conv1)

        conv2 = Conv1D(64, 5, activation="relu", padding="same", name="conv1d_5gram")(x)
        conv2 = MaxPooling1D(pool_size=2, name="maxpool_5gram")(conv2)

        x = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
        x = LSTM(70, dropout=0.2, recurrent_dropout=0.2, name="lstm_layer")(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation="linear", name="q_values")(x)

        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        logger.info("DQN model built successfully")
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(2)
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        minibatch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(
                    target_next[i]
                )

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(
        self,
        env,
        episodes: int = 100,
        batch_size: int = 64,
        target_update_freq: int = 10,
    ):
        history = {"rewards": [], "accuracy": []}
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            correct = 0
            steps = 0

            while True:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay(batch_size)

                total_reward += reward
                correct += 1 if reward > 0 else 0
                steps += 1

                state = next_state
                if done:
                    break

            if episode % target_update_freq == 0:
                self.update_target_model()

            accuracy = correct / steps if steps > 0 else 0
            history["rewards"].append(total_reward)
            history["accuracy"].append(accuracy)
            logger.info(
                f"Episode {episode + 1}/{episodes} - Reward: {total_reward:.2f}, Accuracy: {accuracy:.4f}, Epsilon: {self.epsilon:.4f}"
            )

        self.history = history
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = []
        for state in X_test:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            y_pred.append(action)

        y_pred = np.array(y_pred)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        logger.info("\nDQN Evaluation Results:")
        for metric, value in metrics.items():
            if metric not in ["confusion_matrix", "classification_report"]:
                logger.info(f"{metric.capitalize()}: {value:.4f}")

        return metrics

    def plot_training_history(self):
        if not self.history:
            raise ValueError("No training history available.")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["rewards"], label="Total Reward")
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history["accuracy"], label="Accuracy")
        plt.title("Training Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("dqn_training_history.png")

    def plot_confusion_matrix(self, cm: np.ndarray, normalize: bool = False):
        plt.figure(figsize=(8, 6))
        fmt = ".2f" if normalize else "d"
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("dqn_confusion_matrix.png")

    def plot_roc_pr_curves(self, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = []
        for state in X_test:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)
            y_pred.append(q_values[0][1])

        y_pred = np.array(y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred):.4f})"
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("dqn_roc_curve.png")

        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("dqn_pr_curve.png")

    def save(self, directory: str = "saved_models"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, self.model_name)
        self.model.save(model_path)
        logger.info(f"DQN model saved to {model_path}")

    @classmethod
    def load(cls, directory: str, model_name: str = "durld_dqn") -> "DURLD_DQN":
        model_path = os.path.join(directory, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        dummy_model = cls(model_name=model_name)
        dummy_model.model = tf.keras.models.load_model(model_path)
        dummy_model.target_model = tf.keras.models.load_model(model_path)
        logger.info(f"DQN model loaded from {model_path}")
        return dummy_model

    def predict_url(self, url: str, preprocessor: URLPreprocessor) -> Tuple[float, str]:
        if not preprocessor.validate_url(url):
            logger.warning(f"Invalid URL: {url}")
            return 0.0, "Invalid"
        state = preprocessor.preprocess_urls([url])
        q_values = self.model.predict(state, verbose=0)[0]
        action = np.argmax(q_values)
        proba = q_values[1] / (q_values[0] + q_values[1] + 1e-10)
        label = "Malicious" if action == 1 else "Benign"
        return proba, label
