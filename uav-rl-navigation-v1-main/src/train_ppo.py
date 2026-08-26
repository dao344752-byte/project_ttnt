"""from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from uav2d_env import UAV2DEnv
#from check import UAV2DEnv
from drone_renderer import draw_drone

def make_env():
    env = UAV2DEnv(render_mode=None)
    env = Monitor(env)
    return env
env = DummyVecEnv([make_env])
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device="auto"
)

model.learn(total_timesteps=500_000, progress_bar=True)
model.save("ppo_uav2d_safe")
print("\nTraining finished!")
print("Model saved as: ppo_uav2d_safe.zip")

XXXXXXXXXXX


import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from drone_renderer import draw_drone
from uav2d_env import UAV2DEnv

# CALLBACK (FULL FIXED - NO SB3 LOG DEPENDENCY)
class PPOPlotCallback(BaseCallback):

    def __init__(self):
        super().__init__()

        self.timesteps = []

        self.ep_rew_mean = []
        self.ep_len_mean = []

        self.success_rate = []
        self.collision_rate = []

        # buffers
        self._success_buffer = []
        self._collision_buffer = []
        self._episode_rewards = []
        self._episode_lengths = []

    # STEP
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")

        if infos is None:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue

            # success / collision per episode-end
            if info.get("success", False):
                self._success_buffer.append(1)

            if info.get("collision", False):
                self._collision_buffer.append(1)

            # MONITOR episode info (IMPORTANT)
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        return True

    # ROLLOUT END
    def _on_rollout_end(self):

        self.timesteps.append(self.num_timesteps)
        if len(self._episode_rewards) > 0:
            self.ep_rew_mean.append(np.mean(self._episode_rewards))
        else:
            self.ep_rew_mean.append(0.0)

        if len(self._episode_lengths) > 0:
            self.ep_len_mean.append(np.mean(self._episode_lengths))
        else:
            self.ep_len_mean.append(0.0)

        # ===== SUCCESS RATE =====
        if len(self._success_buffer) > 0:
            self.success_rate.append(np.mean(self._success_buffer))
        else:
            self.success_rate.append(0.0)

        # ===== COLLISION RATE =====
        if len(self._collision_buffer) > 0:
            self.collision_rate.append(np.mean(self._collision_buffer))
        else:
            self.collision_rate.append(0.0)

        # reset buffers
        self._success_buffer = []
        self._collision_buffer = []
        self._episode_rewards = []
        self._episode_lengths = []

# ENV
def make_env():
    env = UAV2DEnv(render_mode=None)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
callback = PPOPlotCallback()

# PPO MODEL
model = PPO(
    policy="MlpPolicy", env=env,
    learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
    gamma=0.995, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1, tensorboard_log="./tensorboard_logs/", device="auto"
)

# TRAIN
model.learn(total_timesteps=1_000_000, progress_bar=True, callback=callback)
model.save("ppo_uav2d_safe")
print("\nTraining finished!")

# CLEAN
def clean(x, t):
    x = np.array(x, dtype=float)
    t = np.array(t, dtype=float)
    if len(x) == 0 or len(t) == 0:
        return [], []
    mask = np.isfinite(x) & np.isfinite(t)
    return t[mask], x[mask]

# PLOT
fig, axs = plt.subplots(2, 2, figsize=(14, 10))


def plot(ax, x, y, title):
    tx, ty = clean(y, x)
    ax.plot(tx, ty)
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.grid(True)


plot(axs[0, 0], callback.timesteps, callback.ep_rew_mean, "Reward Mean")
plot(axs[0, 1], callback.timesteps, callback.ep_len_mean, "Episode Length")
plot(axs[1, 0], callback.timesteps, callback.success_rate, "Success Rate")
plot(axs[1, 1], callback.timesteps, callback.collision_rate, "Collision Rate")

axs[1, 0].set_ylim(0, 1)
axs[1, 1].set_ylim(0, 1)

plt.suptitle("PPO UAV Training Dashboard", fontsize=16)
plt.tight_layout()
plt.show(block=True)
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from drone_renderer import draw_drone
from uav2d_env import UAV2DEnv

# CALLBACK
class PPOPlotCallback(BaseCallback):
    def __init__(self, window_size=200, success_threshold=0.95, collision_threshold=0.02, max_ep_len=55, patience=5, verbose=1):
        super().__init__(verbose)

        # Plot Data
        self.timesteps = []
        self.ep_rew_mean = []
        self.ep_len_mean = []
        self.success_rate = []
        self.collision_rate = []
        self.failure_rate = []

        # Episode Buffers
        self._episode_rewards = []
        self._episode_lengths = []

        self.approx_kl = []
        self.clip_fraction = []

        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []
        self.explained_variance = []
        
        self.std = []

        # Rolling Window
        self.success_history = deque(maxlen=window_size)
        self.collision_history = deque(maxlen=window_size)
        self.failure_history = deque(maxlen=window_size)

        # Early Stop
        self.success_threshold = success_threshold
        self.collision_threshold = collision_threshold
        self.max_ep_len = max_ep_len
        self.patience = patience
        self.good_rollouts = 0
        self.stop_training_flag = False

        # Best Model
        self.best_success_rate = -1.0

    # STEP
    def _on_step(self):
        if self.stop_training_flag:
            return False
        infos = self.locals.get("infos")
        if infos is None:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            # chỉ xử lý episode đã kết thúc
            if "episode" not in info:
                continue
            success = int(info.get("success", False))
            collision = int(info.get("collision", False))
            failure = int((not success) and (not collision))
            self.success_history.append(success)
            self.collision_history.append(collision)
            self.failure_history.append(failure)
            self._episode_rewards.append(info["episode"]["r"])
            self._episode_lengths.append(info["episode"]["l"])
        return True

    # ROLLOUT END
    def _on_rollout_end(self):
        self.timesteps.append(self.num_timesteps)

        # Reward Mean
        reward_mean = (
            np.mean(self._episode_rewards)
            if len(self._episode_rewards) > 0
            else 0.0
        )

        self.ep_rew_mean.append(reward_mean)

        # Episode Length Mean
        ep_len_mean = (
            np.mean(self._episode_lengths)
            if len(self._episode_lengths) > 0
            else 0.0
        )

        self.ep_len_mean.append(ep_len_mean)

        # Rates
        sr = (
            np.mean(self.success_history)
            if len(self.success_history) > 0
            else 0.0
        )

        cr = (
            np.mean(self.collision_history)
            if len(self.collision_history) > 0
            else 0.0
        )

        fr = (
            np.mean(self.failure_history)
            if len(self.failure_history) > 0
            else 0.0
        )

        self.success_rate.append(sr)
        self.collision_rate.append(cr)
        self.failure_rate.append(fr)

        # Save Best Model
        if sr > self.best_success_rate:
            self.best_success_rate = sr
            self.model.save("ppo_uav_best")
            print(
                f"[BEST MODEL] "
                f"SR={sr:.3f}"
            )

        # Console
        print(
            f"[{self.num_timesteps:>8}] "
            f"SR={sr:.3f} "
            f"CR={cr:.3f} "
            f"FR={fr:.3f} "
            f"R={reward_mean:.2f} "
            f"L={ep_len_mean:.1f}"
        )

        # Early Stop
        if (sr >= self.success_threshold and cr <= self.collision_threshold and ep_len_mean <= self.max_ep_len):
            self.good_rollouts += 1
            print(
                f"EarlyStopCheck "
                f"{self.good_rollouts}/{self.patience}"
            )
            if self.good_rollouts >= self.patience:
                print("\n")
                print("=" * 60)
                print("EARLY STOP TRIGGERED")
                print(f"Success Rate  : {sr:.3f}")
                print(f"Collision Rate: {cr:.3f}")
                print(f"Failure Rate  : {fr:.3f}")
                print(f"Episode Length: {ep_len_mean:.1f}")
                print("=" * 60)
                print("\n")
                self.stop_training_flag = True
        else:
            self.good_rollouts = 0

        # Reset rollout buffers
        self._episode_rewards.clear()
        self._episode_lengths.clear()
        

        # bonus plot
        logs = self.model.logger.name_to_value
        def get(key):
            return logs.get(key, np.nan)
        self.approx_kl.append(get("train/approx_kl"))
        self.clip_fraction.append(get("train/clip_fraction"))
        self.policy_loss.append(get("train/policy_gradient_loss"))
        self.value_loss.append(get("train/value_loss"))
        self.entropy_loss.append(get("train/entropy_loss"))
        self.explained_variance.append(get("train/explained_variance"))
        self.std.append(get("train/std"))

        return True

# ENV
def make_env():
    env = UAV2DEnv(render_mode=None)
    env = Monitor(env)
    return env
env = DummyVecEnv([make_env])
callback = PPOPlotCallback( window_size=200, success_threshold=0.95, collision_threshold=0.02, max_ep_len=55, patience=5)

# PPO MODEL
model = PPO(
    policy="MlpPolicy", env=env,
    learning_rate=3e-4,
    n_steps=2048, batch_size=256, n_epochs=10,
    gamma=0.995, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1, tensorboard_log="./tensorboard_logs/", device="auto"
)

# TRAIN
model.learn(total_timesteps= 1_000_000, callback=callback, progress_bar=True)
model.save("ppo_uav2d_safe")
print("\nTraining Finished!")

# PLOT
def clean(x, t):
    x = np.array(x, dtype=float)
    t = np.array(t, dtype=float)
    if len(x) == 0:
        return [], []
    mask = np.isfinite(x) & np.isfinite(t)
    return t[mask], x[mask]

def create_plot(x, y, title, ylim=None):
    tx, ty = clean(y, x)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(tx, ty)
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.grid(True)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
"""
create_plot(callback.timesteps, callback.ep_rew_mean, "Reward Mean")
create_plot(callback.timesteps, callback.ep_len_mean, "Episode Length")
create_plot(callback.timesteps, callback.success_rate, "Success Rate", ylim=(0, 1))
create_plot(callback.timesteps, callback.collision_rate, "Collision Rate", ylim=(0, 1))
create_plot(callback.timesteps, callback.failure_rate, "Failure Rate", ylim=(0, 1))
"""
create_plot(callback.timesteps, callback.approx_kl, "Approx KL")
create_plot(callback.timesteps, callback.clip_fraction, "Clip Fraction")
create_plot(callback.timesteps, callback.entropy_loss, "Entropy Loss")
create_plot(callback.timesteps, callback.value_loss, "Value Loss")
create_plot(callback.timesteps, callback.explained_variance, "Explained Variance")
create_plot(callback.timesteps, callback.std, "Policy Std")
plt.show(block=True)