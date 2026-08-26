"""
from uav2d_env import UAV2DEnv
import numpy as np
import time

env = UAV2DEnv(render_mode="human")

obs, info = env.reset()

for _ in range(300):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.02)

env.close()
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from drone_renderer import (draw_drone, draw_goal, draw_obstacle, draw_danger_zone)
from matplotlib.lines import Line2D

class UAV2DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window_closed = False
        self.dt = 0.2
        self.max_speed = 0.75
        self.max_acceleration = 1.2
        self.velocity_damping = 0.96
        self.world_min = 0.0
        self.world_max = 10.0
        self.world_size = (self.world_max - self.world_min)
        self.num_obstacles = 3
        self.obstacle_radius = 0.4
        self.danger_distance = 1.4
        self.obstacles = []
        self.goal_radius = 0.5
        self.max_steps = 250
        self.step_count = 0
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.prev_vel = np.zeros(2, dtype=np.float32)
        self.prev_dist_to_goal = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.no_progress_steps = 0
        self.trajectory = []
        self.rng = np.random.default_rng()
        self.action_space = spaces.Box( low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        self.fig = None
        self.ax = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.window_closed = False
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.no_progress_steps = 0
        self.prev_action[:] = 0.0
        self.obstacles = []

        for _ in range(self.num_obstacles):
            while True:
                obs = self.rng.uniform( 1.5, 8.5, size=2).astype(np.float32)
                valid = True
                for old_obs in self.obstacles:
                    d = np.linalg.norm(obs - old_obs)
                    if d < 2.2:
                        valid = False
                        break
                if valid:
                    self.obstacles.append(obs)
                    break
        while True:
            self.pos = self.rng.uniform(1.0, 9.0, size=2).astype(np.float32)
            valid = True
            for obs in self.obstacles:
                d = np.linalg.norm(self.pos - obs)
                if d < 2.0:
                    valid = False
                    break
            if valid:
                break
        while True:
            self.goal = self.rng.uniform(1.0, 9.0, size=2).astype(np.float32)
            valid = True
            for obs in self.obstacles:
                d = np.linalg.norm(self.goal - obs)
                if d < 2.0:
                    valid = False
                    break
            if np.linalg.norm(self.goal - self.pos) < 4.0:
                valid = False
            if valid:
                break
        self.vel[:] = 0.0
        self.prev_vel[:] = 0.0
        self.prev_dist_to_goal = np.linalg.norm(self.goal - self.pos)
        self.trajectory = [self.pos.copy()]
        if self.render_mode == "human":
            self._render()
        return self._get_obs(), {}
    def step(self, action):
        self.step_count += 1

    # =========================
    # ACTION + DYNAMICS
    # =========================
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        acceleration = action * self.max_acceleration

        self.vel += acceleration * self.dt
        self.vel *= self.velocity_damping

        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
            speed = self.max_speed

    # =========================
    # TURN PENALTY
    # =========================
        turn_penalty = 0.0
        if np.linalg.norm(self.prev_vel) > 1e-6 and speed > 1e-6:
            v1 = self.prev_vel / (np.linalg.norm(self.prev_vel) + 1e-6)
            v2 = self.vel / (speed + 1e-6)
            cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
            turn_penalty = 1.0 - cos_theta

    # =========================
    # FUTURE RISK
    # =========================
        future_pos = self.pos + self.vel * 8.0
        future_risk = 0.0

        for obs in self.obstacles:
            future_dist = np.linalg.norm(future_pos - obs)
            if future_dist < self.danger_distance:
                future_risk += (self.danger_distance - future_dist)

    # =========================
    # WALL SOFT BOUNDARY
    # =========================
        margin = 0.4
        if future_pos[0] < margin:
            self.vel[0] *= 0.3
        if future_pos[0] > self.world_max - margin:
            self.vel[0] *= 0.3
        if future_pos[1] < margin:
            self.vel[1] *= 0.3
        if future_pos[1] > self.world_max - margin:
            self.vel[1] *= 0.3

    # =========================
    # UPDATE POSITION
    # =========================
        self.pos += self.vel * self.dt
        self.pos = np.clip(self.pos, self.world_min, self.world_max)

        self.trajectory.append(self.pos.copy())

    # =========================
    # DISTANCES
    # =========================
        dist_to_goal = np.linalg.norm(self.goal - self.pos)

        dists = np.array([np.linalg.norm(self.pos - obs) for obs in self.obstacles])
        sorted_idx = np.argsort(dists)
        nearest_dist = dists[sorted_idx[0]]

    # =========================
    # CORRIDOR CENTER (KEY FIX)
    # =========================
        corridor_reward = 0.0
        corridor_center_reward = 0.0

        if len(self.obstacles) >= 2:
            o1 = self.obstacles[sorted_idx[0]]
            o2 = self.obstacles[sorted_idx[1]]

            center = (o1 + o2) / 2.0
            to_center = center - self.pos

            dist_center = np.linalg.norm(to_center)

            if dist_center > 1e-6:
                dir_center = to_center / dist_center
            else:
                dir_center = np.zeros_like(to_center)

            if speed > 1e-6:
                heading = self.vel / speed
            else:
                heading = np.zeros_like(self.vel)

            alignment = np.dot(dir_center, heading)

            corridor_center_reward = alignment * np.exp(-dist_center / 5.0)

            gap = abs(dists[sorted_idx[0]] - dists[sorted_idx[1]])
            corridor_reward = np.exp(-gap)
        

    # =========================
    # PROGRESS
    # =========================
        progress = self.prev_dist_to_goal - dist_to_goal

    # =========================
    # GOAL DIRECTION ALIGNMENT
    # =========================
        goal_vec = self.goal - self.pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-6)

        if speed > 1e-6:
            heading = self.vel / speed
            alignment_goal = np.dot(goal_dir, heading)
        else:
            alignment_goal = 0.0

    # =========================
    # SPEED CONTROL
    # =========================
        speed_reward = 0.05 * speed * np.exp(-dist_to_goal / 3.0)

    # =========================
    # TURN / ACTION SMOOTHNESS
    # =========================
        action_change = np.linalg.norm(action - self.prev_action)

    # =========================
    # FUTURE RISK PENALTY
    # =========================
        risk_penalty = future_risk * 15.0

    # =========================
    # WALL PENALTY
    # =========================
        wall_margin = 0.8
        wall_dist = min( self.pos[0], self.pos[1], self.world_max - self.pos[0], self.world_max - self.pos[1] )

        wall_penalty = 0.0
        if wall_dist < wall_margin:
            wall_penalty = ((wall_margin - wall_dist) ** 2) * 30.0

    # =========================
    # REWARD
    # =========================
        reward = 0.0

    # corridor guidance (IMPORTANT)  

        reward += 2.5 * corridor_center_reward
        reward += 1.0 * corridor_reward

    # goal progress
        reward += np.clip(progress, -0.1, 0.1) * 15.0
        reward += 2.0 * np.exp(-dist_to_goal)

    # alignment to goal
        reward += 1.0 * alignment_goal 

    # shaping penalties
        reward -= 0.03
        reward -= turn_penalty * 0.1
        reward -= action_change * 0.25
        reward -= risk_penalty
        reward -= wall_penalty

        reward += speed_reward
        
        near_goal = np.exp(-dist_to_goal)
        reward += 3.0 * near_goal * alignment_goal

    # obstacle proximity penalty (local safety)
        for obs in self.obstacles:
            d = np.linalg.norm(self.pos - obs)
            if d < self.danger_distance:
                ratio = ((self.danger_distance - d) / self.danger_distance) 
                reward -= 40*ratio

    # =========================
    # STUCK DETECTION
    # =========================
        if progress < 0.002:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

    # =========================
    # TERMINATION FLAGS
    # =========================
        terminated = False
        truncated = False
        is_success = False
        is_collision = False

        for obs in self.obstacles:
            d = np.linalg.norm(self.pos - obs)
            if d < self.obstacle_radius + 0.2:
                reward -= 50

    # COLLISION
        if nearest_dist < self.obstacle_radius:
            reward -= 500.0
            terminated = True
            is_collision = True
            print("Collision!")

    # GOAL
        if dist_to_goal < self.goal_radius:
            reward += 200.0
            reward += (self.max_steps - self.step_count)
            terminated = True
            is_success = True
            print("Goal reached!")

    # STUCK
        if self.no_progress_steps > 40:
            reward -= 50.0
            truncated = True

    # TIME LIMIT
        if self.step_count >= self.max_steps and not terminated:
            reward -= 80.0
            truncated = True

    # =========================
    # UPDATE MEMORY
    # =========================
        self.prev_dist_to_goal = dist_to_goal
        self.prev_vel = self.vel.copy()
        self.prev_action = action.copy()

    # =========================
    # RENDER
    # =========================
        if self.render_mode == "human":
            self._render()
        # print(f"dist={dist_to_goal:.3f} " f"speed={speed:.3f} "f"corridor={corridor_center_reward:.3f}")
        info = {
        "dist_goal": float(dist_to_goal),
        "nearest_obstacle": float(nearest_dist),
        "success": is_success,
        "collision": is_collision,
        "episode_step": self.step_count,
        "reward": reward
    }
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):

        # GOAL VECTOR
        goal_vec = self.goal - self.pos
        dist_goal = np.linalg.norm(goal_vec)
        goal_dir = (goal_vec / (dist_goal + 1e-6))

        # HEADING
        speed = np.linalg.norm(self.vel)

        if speed > 1e-6:
            heading = self.vel / speed
        else:
            heading = np.zeros(2)

        # WALL DISTANCES
        wall_left = (self.pos[0] / self.world_size)
        wall_right = ((self.world_max - self.pos[0]) / self.world_size)
        wall_bottom = (self.pos[1] / self.world_size)
        wall_top = ((self.world_max - self.pos[1]) / self.world_size)

        # NEAREST OBSTACLES
        obstacle_info = []
        for obs in self.obstacles:
            vec = obs - self.pos
            dist = np.linalg.norm(vec)
            direction = (vec / (dist + 1e-6))
            obstacle_info.append((dist, direction))
        obstacle_info.sort(key=lambda x: x[0])
        obs_data = []
        for i in range(3):
            if i < len(obstacle_info):
                dist, direction = obstacle_info[i]
                obs_data.extend([direction[0], direction[1], dist / self.world_size])
            else:
                obs_data.extend([0.0, 0.0, 1.0])

        # FINAL OBSERVATION VECTOR
        obs = np.array([

            # goal direction
            goal_dir[0],
            goal_dir[1],

            # heading direction
            heading[0],
            heading[1],

            # normalized velocity
            self.vel[0] / self.max_speed,
            self.vel[1] / self.max_speed,

            # goal distance
            dist_goal / self.world_size,

            # wall distances
            wall_left,
            wall_right,
            wall_bottom,
            wall_top,

            # obstacle information
            *obs_data

        ], dtype=np.float32)

        return obs

    # RENDER
    def _render(self):

        # CREATE WINDOW
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # WINDOW CLOSED
        if not plt.fignum_exists(self.fig.number):
            self.window_closed = True
            return

        # CLEAR AXIS
        self.ax.clear()

        # DRAW OBSTACLES
        for obs in self.obstacles:
            draw_danger_zone(self.ax, obs, self.danger_distance)
            draw_obstacle(self.ax, obs, self.obstacle_radius)

        # DRAW GOAL
        draw_goal(self.ax, self.goal)

        # DRAW DRONE
        draw_drone( ax=self.ax, pos=self.pos, vel=self.vel, scale=1.1)

        # DRAW TRAJECTORY
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], color="#ff7f00", linewidth=2.5, zorder=3)
   
        # AXIS SETTINGS
        self.ax.set_xlim( self.world_min, self.world_max)
        self.ax.set_ylim( self.world_min, self.world_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True,linestyle="-",alpha=0.25)
        self.ax.tick_params( left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax.set_title(
            f"UAV Navigation | "
            f"Step {self.step_count}/{self.max_steps}"
        )

        # UPDATE FIGURE
        plt.draw()
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='UAV (Drone)', markerfacecolor='dodgerblue', markeredgecolor='black', markersize=10),
            Line2D([0], [0], marker='*', color='green', label='Goal', markersize=12, linestyle='None'),
            Line2D([0], [0], marker='o', color='red', label='Obstacle', markersize=12, linestyle='None'),
            Line2D([0], [0], color='#ff7f00', lw=2.5, label='Trajectory'),
            Line2D([0], [0], color='red', lw=1.2, linestyle='--', label='Danger Zone')
        ]
        self.ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        plt.pause(0.001)

    # CLOSE
    def close(self):
        self.window_closed = True
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

# TEST
if __name__ == "__main__":
    env = UAV2DEnv(render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = (terminated or truncated)
    env.close()


