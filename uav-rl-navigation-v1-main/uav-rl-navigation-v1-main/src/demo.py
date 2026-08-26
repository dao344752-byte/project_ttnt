import numpy as np
from stable_baselines3 import PPO
from uav2d_env import UAV2DEnv
from drone_renderer import draw_drone

MODEL_PATH = "ppo_uav2d_safe"
DETERMINISTIC = True
RENDER = True  

env = UAV2DEnv(render_mode="human" if RENDER else None)
model = PPO.load(MODEL_PATH)
print("Model loaded!")
obs, info = env.reset()

episode = 0
ep_reward = 0.0
ep_steps = 0
success = 0
collision = 0
timeout = 0

while True:
    if getattr(env, "window_closed", False):
        break
    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_reward += reward
    ep_steps += 1
    done = terminated or truncated

    if done:
        episode += 1
        reached_goal = info.get("reached_goal", None)
        if reached_goal is None:
            reached_goal = info["dist_goal"] < env.goal_radius
        if reached_goal:
            success += 1
            status = "GOAL"
        elif terminated:
            collision += 1
            status = "COLLISION"
        else:
            timeout += 1
            status = "TIMEOUT"
        total = success + collision + timeout
        success_rate = 100 * success / max(total, 1)

        print(
            f"[EP {episode:04d}] "
            f"{status:10s} | "
            f"Reward: {ep_reward:8.2f} | "
            f"Steps: {ep_steps:4d} | "
            f"SuccessRate: {success_rate:.1f}%"
        )

        # reset episode
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

env.close()
print("Done.")