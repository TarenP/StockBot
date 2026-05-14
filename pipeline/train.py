"""
PPO training loop with:
- Walk-forward cross-validation
- Gradient clipping + entropy bonus
- Checkpoint saving on best val Sharpe
- Periodic resume checkpoints (every N steps) — survives Ctrl+C
- Auto-resume: skips completed folds, continues mid-fold from last saved step
"""

import os
import logging
import time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pipeline.environment import PortfolioEnv
from pipeline.features import FEATURE_COLS
from pipeline.model import PortfolioTransformer
from pipeline.policy_diagnostics import average_metric_dicts, weight_concentration_metrics

logger = logging.getLogger(__name__)


# ── PPO hyperparameters ──────────────────────────────────────────────────────

PPO_CFG = dict(

    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    entropy_coef    = 0.01,
    value_coef      = 0.5,
    max_grad_norm   = 0.5,
    ppo_epochs      = 4,
    batch_size      = 64,
    rollout_steps   = 256,
    total_steps     = 500_000,
    save_every      = 5_000,    # write a resume checkpoint every N steps
)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _resume_path(save_dir: str, fold_idx: int) -> str:
    return os.path.join(save_dir, f"resume_fold{fold_idx}.pt")

def _best_path(save_dir: str, fold_idx: int) -> str:
    return os.path.join(save_dir, f"best_fold{fold_idx}.pt")

def _done_path(save_dir: str, fold_idx: int) -> str:
    """Marker file written when a fold completes fully."""
    return os.path.join(save_dir, f"fold{fold_idx}.done")

def fold_is_complete(save_dir: str, fold_idx: int) -> bool:
    return os.path.exists(_done_path(save_dir, fold_idx))

def _save_resume(path, model, optimizer, scheduler, model_cfg,
                 fold_idx, steps_done, best_val_sharpe, best_val_return,
                 training_mode: str = "standard",
                 total_steps_requested: int | None = None,
                 feature_cols: list[str] | None = None,
                 asset_list: list[str] | None = None):
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_cfg":       model_cfg,
        "fold":            fold_idx,
        "steps_done":      steps_done,
        "steps":           steps_done,
        "best_val_sharpe": best_val_sharpe,
        "best_val_return": best_val_return,
        "training_mode":   training_mode,
        "total_steps":     total_steps_requested,
        "total_steps_requested": total_steps_requested,
        "feature_cols":    list(feature_cols or []),
        "n_features":      len(feature_cols or []),
        "asset_list":      list(asset_list or []),
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }, path)

def _load_resume(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def log_memory(label: str):
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        logger.info("Memory %s: RAM RSS %.2f GB", label, rss)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            logger.info(
                "Memory %s: CUDA allocated %.2f GB reserved %.2f GB",
                label,
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.memory_reserved() / 1024**3,
            )
    except Exception:
        pass


def _frame_date_range(df):
    dates = sorted(df.index.get_level_values("date").unique())
    if not dates:
        return "empty", "empty"
    return pd_timestamp_date(dates[0]), pd_timestamp_date(dates[-1])


def pd_timestamp_date(value):
    try:
        return value.date()
    except AttributeError:
        return value


def _log_fold_frame_diagnostics(fold_idx, df_train, df_val, asset_list, train_env, val_env, feature_cols):
    train_start, train_end = _frame_date_range(df_train)
    val_start, val_end = _frame_date_range(df_val)
    train_tickers = df_train.index.get_level_values("ticker").nunique()
    val_tickers = df_val.index.get_level_values("ticker").nunique()
    logger.info(
        "[fold %d] train dates = %s to %s | validation dates = %s to %s",
        fold_idx,
        train_start,
        train_end,
        val_start,
        val_end,
    )
    logger.info(
        "[fold %d] train rows = %d tickers = %d | validation rows = %d tickers = %d | asset_list = %d | features = %d",
        fold_idx,
        len(df_train),
        train_tickers,
        len(df_val),
        val_tickers,
        len(asset_list),
        len(feature_cols),
    )
    logger.info(
        "[fold %d] train price tensor shape = %s | validation price tensor shape = %s",
        fold_idx,
        tuple(train_env.price_arr.shape),
        tuple(val_env.price_arr.shape),
    )
    if train_tickers != len(asset_list) or val_tickers != len(asset_list):
        logger.warning(
            "[fold %d] ticker/asset alignment mismatch: train_tickers=%d val_tickers=%d asset_list=%d",
            fold_idx,
            train_tickers,
            val_tickers,
            len(asset_list),
        )


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity / np.maximum(peaks, 1e-12)) - 1.0
    return float(drawdowns.min())


# ── GAE advantage estimation ─────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    last_adv   = 0.0
    for t in reversed(range(len(rewards))):
        next_val  = values[t + 1] if t + 1 < len(values) else 0.0
        delta     = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_adv  = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:len(rewards)]
    return advantages, returns


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(env, model, device, n_steps):
    obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist   = torch.distributions.Dirichlet(F.softplus(logits) + 1e-6)
            action = dist.sample()
            logp   = dist.log_prob(action)

        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action_np)
        logp_list.append(logp.item())
        rew_list.append(reward)
        val_list.append(value.item())
        done_list.append(float(done))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, last_val = model(obs_t)
    val_list.append(last_val.item())

    return (
        np.array(obs_list,  dtype=np.float32),
        np.array(act_list,  dtype=np.float32),
        np.array(logp_list, dtype=np.float32),
        np.array(rew_list,  dtype=np.float32),
        np.array(val_list,  dtype=np.float32),
        np.array(done_list, dtype=np.float32),
    )


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(model, optimizer, obs_b, act_b, logp_old_b, adv_b, ret_b, cfg, device):
    obs_t    = torch.tensor(obs_b,      dtype=torch.float32, device=device)
    act_t    = torch.tensor(act_b,      dtype=torch.float32, device=device)
    logp_old = torch.tensor(logp_old_b, dtype=torch.float32, device=device)
    adv_t    = torch.tensor(adv_b,      dtype=torch.float32, device=device)
    ret_t    = torch.tensor(ret_b,      dtype=torch.float32, device=device)

    logits, values = model(obs_t)
    dist    = torch.distributions.Dirichlet(F.softplus(logits) + 1e-6)
    logp    = dist.log_prob(act_t)
    entropy = dist.entropy().mean()

    ratio  = torch.exp(logp - logp_old)
    adv_n  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    surr1  = ratio * adv_n
    surr2  = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_n

    actor_loss  = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(values.squeeze(-1), ret_t)
    loss        = actor_loss + cfg["value_coef"] * critic_loss - cfg["entropy_coef"] * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
    optimizer.step()

    return {
        "loss":        loss.item(),
        "actor_loss":  actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy":     entropy.item(),
    }


# ── Main training function ────────────────────────────────────────────────────

def train_fold(
    df_train,
    df_val,
    asset_list: list[str],
    fold_idx: int,
    cfg: dict = PPO_CFG,
    model_cfg: dict = None,
    save_dir: str = "models",
    device: torch.device = None,
    seed: int = 42,
    pretrained_state: dict = None,
    top_n: int = 500,
    force_restart: bool = False,
    shortlist_universe: list[str] = None,
    curriculum: bool = False,
    training_mode: str = "standard",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)

    # Use shortlist universe when provided (overrides asset_list)
    if shortlist_universe is not None and len(shortlist_universe) > 0:
        asset_list = shortlist_universe
        tqdm.write(f"  Using shortlist universe: {len(asset_list)} tickers")

    feature_cols = [col for col in FEATURE_COLS if col in df_train.columns]
    if not feature_cols:
        feature_cols = [
            col for col in df_train.columns
            if col not in {"close", "close_raw", "volume"}
        ]
    if not feature_cols:
        raise ValueError("No feature columns available for PPO training.")
    n_features = len(feature_cols)
    lookback   = 20

    if model_cfg is None:
        model_cfg = dict(
            n_assets          = len(asset_list),
            n_features        = n_features,
            lookback          = lookback,
            d_model           = 64,
            nhead_temporal    = 4,
            nhead_cross       = 4,
            num_temporal_layers = 2,
            num_cross_layers  = 1,
            dropout           = 0.1,
        )

    model     = PortfolioTransformer(**model_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(cfg["total_steps"] // cfg["rollout_steps"], 1)
    )

    # ── Resume from mid-fold checkpoint if it exists ─────────────────────────
    steps_done      = 0
    best_val_sharpe = -np.inf
    best_val_return = 0.0
    best_ckpt_path  = _best_path(save_dir, fold_idx)

    resume_path = _resume_path(save_dir, fold_idx)
    if os.path.exists(resume_path) and not force_restart:
        tqdm.write(f"  Resuming fold {fold_idx} from {resume_path} ...")
        ckpt = _load_resume(resume_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        model_cfg       = ckpt["model_cfg"]
        steps_done      = ckpt["steps_done"]
        best_val_sharpe = ckpt["best_val_sharpe"]
        best_val_return = ckpt["best_val_return"]
        tqdm.write(f"  Resumed at step {steps_done:,}  "
                   f"(best val Sharpe so far: {best_val_sharpe:.3f})")
    elif pretrained_state is not None:
        try:
            model.load_state_dict(pretrained_state, strict=False)
            tqdm.write("  Loaded pretrained weights for fine-tuning.")
        except Exception as e:
            tqdm.write(f"  Warning: could not load pretrained weights ({e}).")
    elif force_restart:
        tqdm.write(f"  Force restart enabled for fold {fold_idx} - ignoring old resume state.")

    log_memory(f"[fold {fold_idx}] before training env build")
    train_env = PortfolioEnv(
        df_train,
        asset_list,
        lookback=lookback,
        feature_cols=feature_cols,
    )
    val_env = PortfolioEnv(
        df_val,
        asset_list,
        lookback=lookback,
        feature_cols=feature_cols,
    )
    _log_fold_frame_diagnostics(
        fold_idx,
        df_train,
        df_val,
        asset_list,
        train_env,
        val_env,
        feature_cols,
    )
    log_memory(f"[fold {fold_idx}] before training")

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Fold {fold_idx} | device={device} | "
               f"assets={len(asset_list)} | features={n_features} | "
               f"start_step={steps_done:,}/{cfg['total_steps']:,}")
    tqdm.write(f"{'='*60}")

    save_every = cfg.get("save_every", 5_000)
    steps_since_save = 0

    # ── Curriculum setup ──────────────────────────────────────────────────────
    curriculum_switch_step = cfg["total_steps"] // 2
    using_small_universe = False
    if curriculum and len(asset_list) > 20:
        small_asset_list = asset_list[:20]
        train_env = PortfolioEnv(
            df_train,
            small_asset_list,
            lookback=lookback,
            feature_cols=feature_cols,
        )
        using_small_universe = True
        tqdm.write(
            f"  Curriculum enabled: starting with {len(small_asset_list)} assets, "
            f"expanding to {len(asset_list)} at step {curriculum_switch_step:,}"
        )
    elif curriculum:
        tqdm.write(
            f"  Curriculum enabled but asset_list has only {len(asset_list)} assets "
            f"(<= 20) — using full list from start"
        )

    pbar = tqdm(
        total=cfg["total_steps"],
        initial=steps_done,
        desc=f"Fold {fold_idx}",
        unit="step",
        colour="blue",
        dynamic_ncols=True,
    )

    try:
        while steps_done < cfg["total_steps"]:
            obs_b, act_b, logp_b, rew_b, val_b, done_b = collect_rollout(
                train_env, model, device, cfg["rollout_steps"]
            )
            adv_b, ret_b = compute_gae(
                rew_b, val_b, done_b, cfg["gamma"], cfg["gae_lambda"]
            )

            n   = len(obs_b)
            idx = np.arange(n)
            metrics = {}
            for _ in range(cfg["ppo_epochs"]):
                np.random.shuffle(idx)
                for start in range(0, n, cfg["batch_size"]):
                    mb = idx[start:start + cfg["batch_size"]]
                    metrics = ppo_update(
                        model, optimizer,
                        obs_b[mb], act_b[mb], logp_b[mb],
                        adv_b[mb], ret_b[mb],
                        cfg, device,
                    )

            scheduler.step()
            steps_done      += cfg["rollout_steps"]
            steps_since_save += cfg["rollout_steps"]
            pbar.update(cfg["rollout_steps"])

            # ── Curriculum: expand universe at midpoint ───────────────────────
            if curriculum and using_small_universe and steps_done >= curriculum_switch_step:
                train_env = PortfolioEnv(
                    df_train,
                    asset_list,
                    lookback=lookback,
                    feature_cols=feature_cols,
                )
                using_small_universe = False
                tqdm.write(
                    f"  Curriculum: expanded universe to {len(asset_list)} assets "
                    f"at step {steps_done:,}"
                )

            val_metrics = evaluate_diagnostics(model, val_env, device)
            val_sharpe = val_metrics["sharpe"]
            val_return = val_metrics["total_return"]

            pbar.set_postfix(
                loss       = f"{metrics.get('loss', 0):.3f}",
                entropy    = f"{metrics.get('entropy', 0):.3f}",
                val_sharpe = f"{val_sharpe:.3f}",
                val_ret    = f"{val_return:.1%}",
            )
            logger.info(
                "[fold %d] validation step=%d fresh=true return=%.6f benchmark_return=%.6f "
                "max_drawdown=%.6f turnover=%.6f sharpe=%.6f mean=%.8f std=%.8f n=%d",
                fold_idx,
                steps_done,
                val_metrics["total_return"],
                val_metrics["benchmark_return"],
                val_metrics["max_drawdown"],
                val_metrics["turnover"],
                val_metrics["sharpe"],
                val_metrics["mean_return"],
                val_metrics["std_return"],
                val_metrics["n_returns"],
            )
            logger.info(
                "[fold %d] validation concentration step=%d avg_max_weight=%.6f "
                "avg_top10=%.6f avg_top20=%.6f avg_entropy=%.6f "
                "avg_effective_positions=%.2f avg_cash=%.6f avg_nonzero=%.2f "
                "avg_reward=%.8f avg_excess=%.8f avg_turnover_penalty=%.8f "
                "avg_drawdown_penalty=%.8f avg_entropy_term=%.8f",
                fold_idx,
                steps_done,
                val_metrics.get("avg_max_weight", 0.0),
                val_metrics.get("avg_top_10_weight_sum", 0.0),
                val_metrics.get("avg_top_20_weight_sum", 0.0),
                val_metrics.get("avg_weight_entropy", 0.0),
                val_metrics.get("avg_effective_number_of_positions", 0.0),
                val_metrics.get("avg_cash_weight", 0.0),
                val_metrics.get("avg_nonzero_positions", 0.0),
                val_metrics.get("avg_reward", 0.0),
                val_metrics.get("avg_excess_return_component", 0.0),
                val_metrics.get("avg_turnover_penalty", 0.0),
                val_metrics.get("avg_drawdown_penalty", 0.0),
                val_metrics.get("avg_entropy_term", 0.0),
            )

            # ── Save best checkpoint ─────────────────────────────────────────
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_val_return = val_return
                torch.save({
                    "model_state": model.state_dict(),
                    "model_cfg":   model_cfg,
                    "fold":        fold_idx,
                    "steps":       steps_done,
                    "total_steps": cfg["total_steps"],
                    "total_steps_requested": cfg["total_steps"],
                    "val_sharpe":  val_sharpe,
                    "val_return":  val_return,
                    "top_n":       top_n,
                    "asset_list":  asset_list,
                    "feature_cols": feature_cols,
                    "n_features":  n_features,
                    "created_at":  datetime.now(timezone.utc).isoformat(),
                    "training_mode": training_mode,
                    "validation_metrics": val_metrics,
                }, best_ckpt_path)
                tqdm.write(
                    f"  ✓ Step {steps_done:,} — new best  "
                    f"(sharpe={val_sharpe:.3f}, ret={val_return:.2%})"
                )

            # ── Periodic resume checkpoint ───────────────────────────────────
            if steps_since_save >= save_every:
                _save_resume(
                    resume_path, model, optimizer, scheduler, model_cfg,
                    fold_idx, steps_done, best_val_sharpe, best_val_return,
                    training_mode=training_mode,
                    total_steps_requested=cfg["total_steps"],
                    feature_cols=feature_cols,
                    asset_list=asset_list,
                )
                steps_since_save = 0

    except KeyboardInterrupt:
        # Ctrl+C — save immediately so nothing is lost
        tqdm.write(f"\n  Interrupted! Saving resume checkpoint for fold {fold_idx}...")
        _save_resume(
            resume_path, model, optimizer, scheduler, model_cfg,
            fold_idx, steps_done, best_val_sharpe, best_val_return,
            training_mode=training_mode,
            total_steps_requested=cfg["total_steps"],
            feature_cols=feature_cols,
            asset_list=asset_list,
        )
        tqdm.write(f"  Saved to {resume_path}. Re-run the same command to continue.")
        pbar.close()
        raise   # re-raise so the outer loop also exits cleanly

    pbar.close()

    # ── Mark fold as complete and clean up resume checkpoint ─────────────────
    if training_mode == "memory_light":
        _save_resume(
            resume_path, model, optimizer, scheduler, model_cfg,
            fold_idx, steps_done, best_val_sharpe, best_val_return,
            training_mode=training_mode,
            total_steps_requested=cfg["total_steps"],
            feature_cols=feature_cols,
            asset_list=asset_list,
        )
    open(_done_path(save_dir, fold_idx), "w").close()
    if os.path.exists(resume_path) and training_mode != "memory_light":
        os.remove(resume_path)

    log_memory(f"[fold {fold_idx}] after training fold")
    tqdm.write(f"\nFold {fold_idx} complete. Best val Sharpe: {best_val_sharpe:.3f}")
    return best_ckpt_path, best_val_sharpe


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_diagnostics(model, env, device) -> dict:
    model.eval()
    obs, _ = env.reset()
    done   = False
    rets   = []
    benchmark_rets = []
    turnovers = []
    rewards = []
    weight_metrics = []
    excess_returns = []
    turnover_penalties = []
    drawdown_penalties = []
    entropy_terms = []
    prev_action = np.zeros(env.n_assets + 1, dtype=np.float32)
    prev_action[-1] = 1.0

    n_dates = max(len(env.dates) - env.lookback - 1, 0)
    pbar = tqdm(total=n_dates, desc="  Evaluating", unit="day",
                leave=False, colour="yellow")

    while not done:
        obs_t   = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            weights = model.get_weights(obs_t)
        action  = weights.squeeze(0).cpu().numpy()
        target_returns = env._get_returns(env.ptr)
        benchmark_ret = float(np.nanmean(target_returns)) if len(target_returns) else 0.0
        turnover = float(np.abs(action - prev_action).sum())
        benchmark_rets.append(benchmark_ret)
        turnovers.append(turnover)
        weight_metrics.append(weight_concentration_metrics(action[:-1], cash_weight=float(action[-1])))
        entropy_terms.append(weight_metrics[-1]["weight_entropy"])
        obs, reward, terminated, truncated, info = env.step(action)
        rets.append(info["port_ret"])
        rewards.append(float(reward))
        excess_returns.append(float(info["port_ret"] - benchmark_ret))
        turnover_penalties.append(float(env.tc * turnover))
        drawdown_penalties.append(0.0)
        prev_action = action.astype(np.float32)
        pbar.update(env.step_size)
        done = terminated or truncated

    pbar.close()
    model.train()

    rets = np.array(rets, dtype=np.float32)
    benchmark_rets = np.array(benchmark_rets, dtype=np.float32)
    avg_weight_metrics = average_metric_dicts(weight_metrics)
    reward_metrics = {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_excess_return_component": float(np.mean(excess_returns)) if excess_returns else 0.0,
        "avg_turnover_penalty": float(np.mean(turnover_penalties)) if turnover_penalties else 0.0,
        "avg_drawdown_penalty": float(np.mean(drawdown_penalties)) if drawdown_penalties else 0.0,
        "avg_entropy_term": float(np.mean(entropy_terms)) if entropy_terms else 0.0,
    }
    if len(rets) < 2:
        return {
            "sharpe": 0.0,
            "total_return": 0.0,
            "benchmark_return": float(np.prod(1 + benchmark_rets) - 1) if len(benchmark_rets) else 0.0,
            "max_drawdown": 0.0,
            "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
            "mean_return": float(rets.mean()) if len(rets) else 0.0,
            "std_return": float(rets.std()) if len(rets) else 0.0,
            "n_returns": int(len(rets)),
        } | avg_weight_metrics | reward_metrics

    mean_r = float(rets.mean())
    std_r = float(rets.std())
    sharpe  = (mean_r / (std_r + 1e-9)) * np.sqrt(252)
    total_r = float(np.prod(1 + rets) - 1)
    benchmark_r = float(np.prod(1 + benchmark_rets) - 1) if len(benchmark_rets) else 0.0
    return {
        "sharpe": float(sharpe),
        "total_return": total_r,
        "benchmark_return": benchmark_r,
        "max_drawdown": _max_drawdown(rets),
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "mean_return": mean_r,
        "std_return": std_r,
        "n_returns": int(len(rets)),
    } | avg_weight_metrics | reward_metrics


def evaluate(model, env, device) -> tuple[float, float]:
    metrics = evaluate_diagnostics(model, env, device)
    return metrics["sharpe"], metrics["total_return"]


# ── Shortlist universe builder ────────────────────────────────────────────────

def build_shortlist_universe(
    df_features,
    screener_model,
    top_n: int = 100,
    device: torch.device = None,
    max_universe: int = 150,
) -> list[str]:
    """
    Run the screener on training data and return the tickers that most
    consistently appear in the top-N shortlist across all dates.

    Rather than taking the union (which grows to nearly the full universe over
    many years), we rank tickers by appearance frequency and return the top
    `max_universe` most consistently high-scoring names. This keeps the RL
    training universe focused and tractable.

    Parameters
    ----------
    df_features : pd.DataFrame
        MultiIndex [date, ticker] feature DataFrame (training split).
    screener_model : TickerScorer
        Trained screener model (from pipeline.screener.load_screener).
    top_n : int
        Number of top candidates to select per date.
    device : torch.device | None
        Inference device. Defaults to CPU.
    max_universe : int
        Maximum number of tickers to return (ranked by appearance frequency).

    Returns
    -------
    list[str]
        Sorted list of up to max_universe tickers most consistently shortlisted.
    """
    if device is None:
        device = torch.device("cpu")

    from pipeline.screener import LOOKBACK, MIN_HISTORY_COVERAGE, _heuristic_scores_from_windows

    feat_cols = getattr(screener_model, "_feature_cols", [])
    if not feat_cols:
        from pipeline.features import FEATURE_COLS as _FC
        feat_cols = [c for c in _FC if c in df_features.columns]

    dates = sorted(df_features.index.get_level_values("date").unique())
    tickers = sorted(df_features.index.get_level_values("ticker").unique())
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n_t = len(tickers)
    n_features = len(feat_cols)

    # Build aligned feature array
    feat_arr = np.full((len(dates), n_t, n_features), np.nan, dtype=np.float32)
    present_mask = np.zeros((len(dates), n_t), dtype=bool)
    date_idx = {d: i for i, d in enumerate(dates)}

    for (date, ticker), row in df_features[feat_cols].iterrows():
        di = date_idx.get(date)
        ti = ticker_idx.get(ticker)
        if di is None or ti is None:
            continue
        feat_arr[di, ti, :] = row.values.astype(np.float32)
        present_mask[di, ti] = True

    feat_arr = np.nan_to_num(np.clip(feat_arr, -5.0, 5.0), nan=0.0)

    # Count how many times each ticker appears in the top-N across all dates
    appearance_counts = np.zeros(n_t, dtype=np.int32)
    screener_model.eval()

    for di in range(LOOKBACK, len(dates)):
        history_coverage = present_mask[di - LOOKBACK:di].mean(axis=0)
        valid_tickers_idx = np.where(history_coverage >= MIN_HISTORY_COVERAGE)[0]
        if len(valid_tickers_idx) == 0:
            continue

        obs_arr = np.array(
            [feat_arr[di - LOOKBACK:di, ti, :] for ti in valid_tickers_idx],
            dtype=np.float32,
        )
        X = torch.tensor(obs_arr, device=device)

        scores = []
        with torch.no_grad():
            for start in range(0, len(X), 512):
                logits = screener_model(X[start:start + 512]).squeeze(1)
                scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        scores_arr = np.array(scores, dtype=np.float32)
        k = min(top_n, len(scores_arr))
        top_local_idx = np.argsort(scores_arr)[-k:]
        for local_i in top_local_idx:
            appearance_counts[valid_tickers_idx[local_i]] += 1

    # Rank by frequency, take top max_universe
    ranked_idx = np.argsort(appearance_counts)[::-1][:max_universe]
    result = sorted([tickers[i] for i in ranked_idx if appearance_counts[i] > 0])

    total_dates = len(dates) - LOOKBACK
    tqdm.write(
        f"  Shortlist universe: {len(result)} tickers "
        f"(top-{max_universe} by frequency in top-{top_n} across {total_dates} dates)"
    )
    return result
