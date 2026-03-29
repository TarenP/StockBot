"""
PPO training loop with:
- Walk-forward cross-validation
- Gradient clipping + entropy bonus
- Checkpoint saving on best val Sharpe
- Periodic resume checkpoints (every N steps) — survives Ctrl+C
- Auto-resume: skips completed folds, continues mid-fold from last saved step
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pipeline.environment import PortfolioEnv
from pipeline.model import PortfolioTransformer


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
    total_steps     = 100_000,
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
                 fold_idx, steps_done, best_val_sharpe, best_val_return):
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_cfg":       model_cfg,
        "fold":            fold_idx,
        "steps_done":      steps_done,
        "best_val_sharpe": best_val_sharpe,
        "best_val_return": best_val_return,
    }, path)

def _load_resume(path, device):
    return torch.load(path, map_location=device, weights_only=False)


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
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)

    n_features = df_train.shape[1]
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
        optimizer, T_max=cfg["total_steps"] // cfg["rollout_steps"]
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

    train_env = PortfolioEnv(df_train, asset_list, lookback=lookback)
    val_env   = PortfolioEnv(df_val,   asset_list, lookback=lookback)

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Fold {fold_idx} | device={device} | "
               f"assets={len(asset_list)} | features={n_features} | "
               f"start_step={steps_done:,}/{cfg['total_steps']:,}")
    tqdm.write(f"{'='*60}")

    save_every = cfg.get("save_every", 5_000)
    steps_since_save = 0

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

            val_sharpe, val_return = evaluate(model, val_env, device)

            pbar.set_postfix(
                loss       = f"{metrics.get('loss', 0):.3f}",
                entropy    = f"{metrics.get('entropy', 0):.3f}",
                val_sharpe = f"{val_sharpe:.3f}",
                val_ret    = f"{val_return:.1%}",
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
                    "val_sharpe":  val_sharpe,
                    "val_return":  val_return,
                    "top_n":       top_n,
                    "asset_list":  asset_list,
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
                )
                steps_since_save = 0

    except KeyboardInterrupt:
        # Ctrl+C — save immediately so nothing is lost
        tqdm.write(f"\n  Interrupted! Saving resume checkpoint for fold {fold_idx}...")
        _save_resume(
            resume_path, model, optimizer, scheduler, model_cfg,
            fold_idx, steps_done, best_val_sharpe, best_val_return,
        )
        tqdm.write(f"  Saved to {resume_path}. Re-run the same command to continue.")
        pbar.close()
        raise   # re-raise so the outer loop also exits cleanly

    pbar.close()

    # ── Mark fold as complete and clean up resume checkpoint ─────────────────
    open(_done_path(save_dir, fold_idx), "w").close()
    if os.path.exists(resume_path):
        os.remove(resume_path)

    tqdm.write(f"\nFold {fold_idx} complete. Best val Sharpe: {best_val_sharpe:.3f}")
    return best_ckpt_path, best_val_sharpe


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, env, device) -> tuple[float, float]:
    model.eval()
    obs, _ = env.reset()
    done   = False
    rets   = []

    n_dates = len(env.dates) - env.lookback
    pbar = tqdm(total=n_dates, desc="  Evaluating", unit="day",
                leave=False, colour="yellow")

    while not done:
        obs_t   = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        weights = model.get_weights(obs_t)
        action  = weights.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        rets.append(info["port_ret"])
        pbar.update(env.step_size)
        done = terminated or truncated

    pbar.close()
    model.train()

    rets = np.array(rets)
    if len(rets) < 2:
        return 0.0, 0.0

    sharpe  = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
    total_r = float(np.prod(1 + rets) - 1)
    return sharpe, total_r
