# portfolio_tf_agent.py
"""
Scaffold for a TensorFlow Agents reinforcement learning agent for portfolio management.
Assumes you have a pandas DataFrame `master_df` with:
    - MultiIndex (date, ticker) or flat index per date
    - Columns for each asset's daily (or desired frequency) return, e.g. 'AAPL_ret', 'MSFT_ret', etc.
    - Feature columns already engineered (technical indicators, sentiment scores, macro context, etc.)

The environment expects:
    * `asset_cols`: list of columns containing asset returns (float values of next‑period returns)
    * `feature_cols`: list of columns providing state features shared across assets

Reward is defined as the portfolio‑level return net of proportional transaction costs.
You can customise risk‑adjusted objectives (e.g., Sharpe or Sortino) in the `_step` method.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network, critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common

# =====================================================
# 1. ENVIRONMENT DEFINITION
# =====================================================
class PortfolioEnv(py_environment.PyEnvironment):
    """Vectorised N‑asset portfolio trading environment."""

    def __init__(
        self,
        df: pd.DataFrame,
        asset_cols: list[str],
        feature_cols: list[str],
        window: int = 30,
        initial_cash: float = 1.0,
        transaction_cost: float = 5e‑4,
    ) -> None:
        super().__init__()
        self._df = df.sort_index()
        self.asset_cols = asset_cols
        self.feature_cols = feature_cols
        self.window = window
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash

        self.num_assets = len(asset_cols)

        # Action: portfolio weights (continuous, sum to 1 after normalisation)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self.num_assets,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name="portfolio_weights",
        )

        # Observation: (window, feature_dim, num_assets) tensor
        obs_shape = (self.window, len(feature_cols), self.num_assets)
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=obs_shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="observation",
        )
        self.reset()

    # ----- Spec getters -----
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # ----- Helper methods -----
    def _get_observation(self):
        # Slice the rolling window for each feature and asset
        window_df = self._df.iloc[self._current_idx - self.window + 1 : self._current_idx + 1]
        obs = np.stack(
            [window_df[f].values.reshape(self.window, self.num_assets) for f in self.feature_cols],
            axis=1,
        ).astype(np.float32)  # shape (window, feature_dim, num_assets)
        return obs

    # ----- PyEnvironment overrides -----
    def _reset(self):
        self._current_idx = self.window - 1  # first valid index
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.full(self.num_assets, 1.0 / self.num_assets, dtype=np.float32)
        return ts.restart(self._get_observation())

    def _step(self, action):
        if self._current_idx >= len(self._df) - 1:
            return self.reset()

        # Normalise and clip weights
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)
        action = action / (np.sum(action) + 1e‑8)

        # Transaction cost on weight change
        tc = self.transaction_cost * np.sum(np.abs(action - self.prev_weights))

        # Next‑period returns vector
        next_idx = self._current_idx + 1
        returns_vec = self._df.iloc[next_idx][self.asset_cols].values.astype(np.float32)
        portfolio_ret = np.dot(action, returns_vec) - tc

        # Update state variables
        self.portfolio_value *= 1.0 + portfolio_ret
        self.prev_weights = action
        self._current_idx = next_idx

        observation = self._get_observation()
        reward = portfolio_ret  # Optional: risk‑adjust here
        discount = 0.99

        if self._current_idx >= len(self._df) - 1:
            return ts.termination(observation, reward)
        return ts.transition(observation, reward, discount)


# =====================================================
# 2. ENVIRONMENT WRAPPERS & UTILITIES
# =====================================================

def load_master_dataframe(path: str) -> pd.DataFrame:
    """Load your master feature‑rich DataFrame."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    raise ValueError("Unsupported file format – use parquet or pickle.")


def create_tf_env(df, asset_cols, feature_cols, window=30):
    py_env = PortfolioEnv(df, asset_cols, feature_cols, window)
    return tf_py_environment.TFPyEnvironment(py_env)


# =====================================================
# 3. TRAINING LOOP (Soft Actor‑Critic)
# =====================================================

def train_agent(
    master_df_path: str,
    asset_cols: list[str],
    feature_cols: list[str],
    window: int = 30,
    num_iterations: int = 500_000,
    collect_steps_per_iteration: int = 1,
    replay_buffer_capacity: int = 100_000,
    batch_size: int = 256,
    log_interval: int = 1_000,
    eval_interval: int = 10_000,
    learning_rate: float = 3e‑4,
):
    # ---------- Load data & build envs ----------
    df = load_master_dataframe(master_df_path)
    train_env = create_tf_env(df, asset_cols, feature_cols, window)
    eval_env = create_tf_env(df, asset_cols, feature_cols, window)

    # ---------- Networks ----------
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(256, 256),
        activation_fn=tf.keras.activations.relu,
    )

    critic_net = critic_network.CriticNetwork(
        (train_env.observation_spec(), train_env.action_spec()),
        observation_fc_layer_params=(256,),
        action_fc_layer_params=None,
        joint_fc_layer_params=(256, 256),
    )

    # ---------- Agent ----------
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = sac_agent.SacAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(learning_rate),
        target_update_tau=5e‑3,
        target_update_period=1,
        gamma=0.99,
        reward_scale_factor=1.0,
        train_step_counter=global_step,
    )
    tf_agent.initialize()

    # ---------- Replay buffer ----------
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
    )

    replay_observer = [replay_buffer.add_batch]
    dataset = (
        replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2, num_parallel_calls=3)
        .prefetch(3)
    )
    iterator = iter(dataset)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration,
    )

    # Make functions graph‑compatible for speed
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    # Initial buffer fill (optional warm‑up)
    for _ in range(1_000):
        collect_driver.run()

    # ---------- Training main loop ----------
    for _ in range(num_iterations):
        collect_driver.run()
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)

        step = tf_agent.train_step_counter.numpy()
        if step % log_interval == 0:
            print(f"Step {step:,} | Loss {train_loss.loss.numpy():.4f}")

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_episodes=5)
            print(f"Step {step:,} | Avg Return {avg_return:.4f}")

    # ---------- Save policy ----------
    checkpoint_dir = "./portfolio_sac_checkpoints"
    tf_policy_saver = tf.compat.v2.saved_model.SaveOptions()
    ckpt = tf.train.Checkpoint(agent=tf_agent)
    ckpt.save(file_prefix=f"{checkpoint_dir}/ckpt")
    print("Training complete – model saved.")


# =====================================================
# 4. EVALUATION HELPER
# =====================================================

def compute_avg_return(environment, policy, num_episodes: int = 10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 1.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return *= 1.0 + time_step.reward.numpy()[0]
        total_return += episode_return - 1.0
    return total_return / num_episodes


# =====================================================
# 5. EXAMPLE CLI ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    master_df_path = "Master.parquet"  # <-- update path
    asset_cols = [
        "AAPL_ret",
        "MSFT_ret",
        "SPY_ret",
    ]  # <-- update with your universe
    feature_cols = [
        "sentiment",
        "sma_10",
        "momentum_5",
        "volatility",
    ]

    train_agent(
        master_df_path,
        asset_cols,
        feature_cols,
        window=30,
        num_iterations=300_000,
    )
