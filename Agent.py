# Portfolio RL Training Template (TorchRL + PyTorch)
# Assumes master_df is a MultiIndex [date, ticker] with clean features

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from gymnasium import Env, spaces
from torchrl.envs import GymWrapper
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import OneHotCategorical, Dirichlet
from torchrl.trainers import PPOTrainer
from torchrl.collectors import SyncDataCollector
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.data.replay_buffers import TensorDictReplayBuffer, SamplerWithoutReplacement
from torchrl.data import BoundedTensorSpec, CompositeSpec

# ========== CONFIG ==========
LOOKBACK = 20
#Top K assets to select
TOP_K = 10
CASH_ASSET = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ENVIRONMENT ==========
class PortfolioEnv(Env):
    def __init__(self, df, asset_list, lookback=LOOKBACK):
        super().__init__()
        self.df = df
        self.asset_list = asset_list
        self.n_assets = len(asset_list)
        self.lookback = lookback
        self.ptr = 0

        feature_dim = df.shape[1] // self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback, self.n_assets, feature_dim), dtype=np.float32)

        self.action_space = spaces.Dict({
            'mask': spaces.MultiBinary(self.n_assets),
            'weights': spaces.Box(0, 1, shape=(self.n_assets + int(CASH_ASSET),))
        })

        self.dates = sorted(list(set(idx[0] for idx in df.index)))
        self.reset()

    def reset(self):
        self.ptr = self.lookback
        self.portfolio_value = 1.0
        return self._get_obs()

    def step(self, action):
        mask = action['mask']
        raw_weights = action['weights'][:self.n_assets]
        weights = (mask * raw_weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        today = self.dates[self.ptr - 1]
        tomorrow = self.dates[self.ptr]
        prices_today = self._get_prices(today)
        prices_tomorrow = self._get_prices(tomorrow)

        rets = (prices_tomorrow / prices_today) - 1.0
        portfolio_return = (weights * rets).sum()
        self.portfolio_value *= (1 + portfolio_return)

        reward = np.log1p(portfolio_return)
        self.ptr += 5
        done = self.ptr >= len(self.dates) - 5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs_dates = self.dates[self.ptr - self.lookback: self.ptr]
        obs = []
        for date in obs_dates:
            slice_df = self.df.loc[date].loc[self.asset_list].values
            obs.append(slice_df)
        return np.stack(obs).astype(np.float32)

    def _get_prices(self, date):
        return self.df.loc[date]['close'].values

# ========== MODEL ==========
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ========== POLICY + VALUE ==========
def build_agent(input_shape, n_assets):
    flat_input = np.prod(input_shape)

    selector_net = MLP(flat_input, n_assets)
    allocator_net = MLP(flat_input + n_assets, n_assets + int(CASH_ASSET))
    value_net = MLP(flat_input, 1)

    selector = ProbabilisticActor(
        module=selector_net,
        in_keys=["observation"],
        out_keys=["mask"],
        distribution_class=OneHotCategorical
    )

    allocator = ProbabilisticActor(
        module=allocator_net,
        in_keys=["observation", "mask"],
        out_keys=["weights"],
        distribution_class=Dirichlet
    )

    value = ValueOperator(module=value_net, in_keys=["observation"])
    return selector, allocator, value

# ========== TRAINING LOOP ==========
def train(df, asset_list):
    env = PortfolioEnv(df, asset_list)
    wrapped_env = GymWrapper(env)

    input_shape = wrapped_env.reset().shape
    selector, allocator, value = build_agent(input_shape, len(asset_list))

    class CombinedPolicy(nn.Module):
        def forward(self, tensordict):
            mask_td = selector(tensordict)
            full_td = allocator(mask_td)
            return full_td

    policy = CombinedPolicy()

    trainer = PPOTrainer(
        env=wrapped_env,
        policy=policy,
        value_network=value,
        lr=3e-4,
        gamma=0.99,
        num_epochs=10,
        frames_per_batch=128,
        total_frames=50_000,
        device=DEVICE,
        logger=get_logger('stdout', exp_name=generate_exp_name("portfolio_rl")),
    )

    trainer.train()

# ========== EVALUATION LOOP ==========
def evaluate_policy(policy, df_test, asset_list):
    env = PortfolioEnv(df_test, asset_list)
    obs = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    
    while not done:
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(DEVICE)
        tensordict = {"observation": obs_tensor}
        
        with torch.no_grad():
            mask_td = policy.selector(tensordict)
            full_td = policy.allocator(mask_td)

        action = {
            'mask': full_td['mask'][0].cpu().numpy(),
            'weights': full_td['weights'][0].cpu().numpy()
        }
        
        obs, reward, done, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)

    final_return = portfolio_values[-1] / portfolio_values[0] - 1
    log_return = np.log(portfolio_values[-1]) - np.log(portfolio_values[0])
    print(f"Final portfolio return: {final_return:.2%}")
    print(f"Log return: {log_return:.4f}")

# ========== ENTRY ==========
# Usage: load your dataframe and call train()
# df = pd.read_parquet("master_dataframe.parquet")
# asset_list = sorted(df.index.get_level_values(1).unique())
# train(df, asset_list)
# ========== ENTRY ==========
df = pd.read_parquet("Master_cleaned.parquet")
asset_list = sorted(df.index.get_level_values(1).unique())


evaluate_policy(trained_policy, df.loc['2019'], asset_list)
