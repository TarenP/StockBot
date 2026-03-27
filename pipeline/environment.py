"""
Portfolio environment with:
- Proper Sharpe-ratio-based reward
- Transaction cost modelling
- NaN-safe observations
- Correct cash handling
"""

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class PortfolioEnv(Env):
    """
    At each step the agent observes a (lookback, n_assets, n_features) window
    and outputs portfolio weights over n_assets + 1 (cash).

    Reward = rolling Sharpe ratio contribution minus transaction costs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        asset_list: list[str],
        lookback: int = 20,
        top_k: int = 10,
        transaction_cost: float = 0.001,   # 10 bps per trade
        sharpe_window: int = 20,           # rolling window for Sharpe reward
        step_size: int = 1,
    ):
        super().__init__()
        self.asset_list       = asset_list
        self.n_assets         = len(asset_list)
        self.lookback         = lookback
        self.top_k            = top_k
        self.tc               = transaction_cost
        self.sharpe_window    = sharpe_window
        self.step_size        = step_size
        self.n_features       = df.shape[1]

        # ── Pivot to [date, ticker] → wide format for fast slicing ──────────
        self.dates = sorted(df.index.get_level_values("date").unique())
        self._build_arrays(df)

        # ── Spaces ──────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(lookback, self.n_assets, self.n_features),
            dtype=np.float32,
        )
        # Weights over assets + cash (softmax applied inside step)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_assets + 1,),
            dtype=np.float32,
        )

        self.reset()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_arrays(self, df: pd.DataFrame):
        """Pre-build numpy arrays indexed by [date_idx, asset_idx, feature_idx]."""
        n_dates  = len(self.dates)
        date_map = {d: i for i, d in enumerate(self.dates)}

        self.feat_arr  = np.zeros((n_dates, self.n_assets, self.n_features), dtype=np.float32)
        self.price_arr = np.zeros((n_dates, self.n_assets), dtype=np.float32)

        asset_map = {a: i for i, a in enumerate(self.asset_list)}

        for (date, ticker), row in df.iterrows():
            if ticker not in asset_map:
                continue
            di = date_map[date]
            ai = asset_map[ticker]
            self.feat_arr[di, ai, :]  = row.values.astype(np.float32)

        # Close prices from the raw parquet are not in df (features only).
        # We store a separate price array loaded externally via set_prices().
        # If not set, we approximate returns from ret_1d feature.
        self._prices_set = False

    def set_prices(self, close_arr: np.ndarray):
        """
        Optionally provide raw close prices array shaped [n_dates, n_assets].
        Enables accurate return calculation instead of using ret_1d feature.
        """
        self.price_arr    = close_arr.astype(np.float32)
        self._prices_set  = True

    def _get_obs(self) -> np.ndarray:
        start = self.ptr - self.lookback
        obs   = self.feat_arr[start:self.ptr]   # (lookback, n_assets, n_features)
        return obs.copy()

    def _get_returns(self, t: int) -> np.ndarray:
        """Daily returns at time index t."""
        if self._prices_set:
            p0 = self.price_arr[t - 1]
            p1 = self.price_arr[t]
            mask = p0 > 0
            rets = np.where(mask, (p1 / (p0 + 1e-9)) - 1.0, 0.0)
        else:
            # Fall back to ret_1d feature (index 0 in FEATURE_COLS)
            rets = self.feat_arr[t, :, 0]
        return rets.astype(np.float32)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr             = self.lookback
        self.portfolio_value = 1.0
        self.weights         = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1]     = 1.0   # start fully in cash
        self.return_history  = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Softmax → valid probability distribution
        action  = np.array(action, dtype=np.float32)
        exp_a   = np.exp(action - action.max())
        new_weights = exp_a / exp_a.sum()

        # Transaction costs: proportional to weight change
        tc_cost = self.tc * np.abs(new_weights - self.weights).sum()

        # Asset returns
        rets = self._get_returns(self.ptr)

        # Portfolio return (exclude cash from rets, cash earns 0)
        asset_weights = new_weights[:-1]
        port_ret      = float((asset_weights * rets).sum()) - tc_cost

        self.portfolio_value *= (1.0 + port_ret)
        self.weights          = new_weights
        self.return_history.append(port_ret)

        # ── Reward: Sharpe contribution ──────────────────────────────────────
        if len(self.return_history) >= self.sharpe_window:
            window = np.array(self.return_history[-self.sharpe_window:])
            mean_r = window.mean()
            std_r  = window.std() + 1e-9
            reward = float(mean_r / std_r)
        else:
            reward = float(port_ret)

        self.ptr += self.step_size
        terminated = self.ptr >= len(self.dates) - 1
        truncated  = False

        return self._get_obs(), reward, terminated, truncated, {
            "portfolio_value": self.portfolio_value,
            "port_ret": port_ret,
        }

    def render(self):
        print(f"Step {self.ptr} | Portfolio value: {self.portfolio_value:.4f}")
