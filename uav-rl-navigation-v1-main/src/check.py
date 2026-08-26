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
        self.num_obstacles = 6
        self.obstacle_radius = 0.4
        self.danger_distance = 1.4
        self.num_rays = 24
        self.radar_range = 3.0
        self.obstacles = []
        self.goal_radius = 0.35
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
        obs_dim = 11 + self.num_rays
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.fig = None
        self.ax = None

    # RESET ENVIRONMENT
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.window_closed = False
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # RESET EPISODE VARIABLES
        self.step_count = 0
        self.no_progress_steps = 0
        self.prev_action[:] = 0.0
        self.obstacles = []

        # GENERATE OBSTACLES
        for _ in range(self.num_obstacles):
            while True:
                obs = self.rng.uniform( 1.5, 8.5, size=2).astype(np.float32)
                valid = True

                # tránh obstacles overlap
                for old_obs in self.obstacles:
                    d = np.linalg.norm(obs - old_obs)
                    if d < 2.2:
                        valid = False
                        break
                if valid:
                    self.obstacles.append(obs)
                    break

        # RANDOM UAV POSITION
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

        # RANDOM GOAL POSITION
        while True:
            self.goal = self.rng.uniform(1.0, 9.0, size=2).astype(np.float32)
            valid = True
            for obs in self.obstacles:
                d = np.linalg.norm(self.goal - obs)
                if d < 2.0:
                    valid = False
                    break

            # goal không quá gần UAV
            if np.linalg.norm(self.goal - self.pos) < 4.0:
                valid = False
            if valid:
                break

        # RESET VELOCITY
        self.vel[:] = 0.0
        self.prev_vel[:] = 0.0

        # INITIAL GOAL DISTANCE
        self.prev_dist_to_goal = np.linalg.norm(self.goal - self.pos)

        # RESET TRAJECTORY
        self.trajectory = [self.pos.copy()]

        # RENDER
        if self.render_mode == "human":
            self._render()
        return self._get_obs(), {}

    # STEP
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        acceleration = (action * self.max_acceleration)
        self.vel += (acceleration * self.dt)
        self.vel *= self.velocity_damping
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        turn_penalty = 0.0
        if ( np.linalg.norm(self.prev_vel) > 1e-6 and np.linalg.norm(self.vel) > 1e-6):
            v1 = self.prev_vel / np.linalg.norm(self.prev_vel)
            v2 = self.vel / np.linalg.norm(self.vel)
            cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
            turn_penalty = 1.0 - cos_theta

        # WALL AVOIDANCE
        future_pos = (self.pos + self.vel * 2.0)
        future_risk = 0.0
        for k in range(1, 6):
            predict_pos = self.pos + self.vel * self.dt * k * 3
            for obs in self.obstacles:
                d = np.linalg.norm(predict_pos - obs)
                if d < self.danger_distance:
                    future_risk += (self.danger_distance - d) / k
        # UPDATE POSITION
        self.pos += (self.vel * self.dt)
        self.pos = np.clip(self.pos, self.world_min, self.world_max)

        # SAVE TRAJECTORY
        self.trajectory.append(self.pos.copy())

        # DISTANCE TO GOAL
        dist_to_goal = np.linalg.norm(self.goal - self.pos)

        # NEAREST OBSTACLE
        nearest_dist = 9999.0
        for obs in self.obstacles:
            d = np.linalg.norm(self.pos - obs)
            if d < nearest_dist:
                nearest_dist = d
        #RADAR
        radar = self._radar_scan()

        # REWARD
        reward = 0.0
        reward -= future_risk * 4.0
        front_radar = np.min(radar[10:14])
        reward -= (1.0 - front_radar) * 2.0

        # CLEARANCE PENALTY
        """
        safe_distance = 2.5
        if nearest_dist < safe_distance:
            clearance_penalty = ((safe_distance - nearest_dist) / safe_distance) ** 2
            reward -= clearance_penalty * 1.5
        """

        # PROGRESS REWARD
        progress = (self.prev_dist_to_goal - dist_to_goal)
        reward += progress * 25.0
        goal_vec = self.goal - self.pos
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
        if np.linalg.norm(self.vel) > 1e-6:
            vel_dir = self.vel / np.linalg.norm(self.vel)
            alignment = np.dot(vel_dir, goal_dir)
            reward += alignment * 0.5

        if dist_to_goal < 2.0:
            reward += 0.3
        if dist_to_goal < 1.0:
            reward += 0.6
        if dist_to_goal < 0.5:
            reward += 1.0

        # TIME PENALTY
        reward -= 0.03

        # SPEED BONUS
        reward += 0.02 * speed
        reward -= turn_penalty * 0.35

        # ACTION SMOOTHNESS
        action_change = np.linalg.norm(action - self.prev_action)
        reward -= action_change * 0.25
        self.prev_action = action.copy()

        # OBSTACLE DANGER PENALTY
        for obs in self.obstacles:
            d = np.linalg.norm(self.pos - obs)
            if d < self.danger_distance:
                penalty = ((self.danger_distance - d) / self.danger_distance) ** 2 
                penalty *= max(0.3, 1.0 - progress * 8)
                reward -= penalty * 5.0

        # WALL PENALTY
        wall_margin = 0.8
        wall_dist = min( self.pos[0], self.pos[1], self.world_max - self.pos[0], self.world_max - self.pos[1])
        if wall_dist < wall_margin:
            reward -= ((wall_margin - wall_dist) ** 2) * 30.0

        # STUCK DETECTION
        if abs(progress) < 0.0001:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        # TERMINATION FLAGS
        terminated = False
        truncated = False
        is_success = False
        is_collision = False

        # COLLISION
        if nearest_dist < self.obstacle_radius:
            reward -= 500.0
            terminated = True
            is_collision = True
            print("Collision!")

        # GOAL REACHED
        if dist_to_goal < self.goal_radius:
            reward += 200.0
            reward += (self.max_steps - self.step_count)
            terminated = True
            is_success = True
            print("Goal reached!")

        # STUCK TERMINATION
        if self.no_progress_steps > 40:
            reward -= 50.0
            truncated = True

        # MAX STEP TERMINATION
        if (self.step_count >= self.max_steps and not terminated):
            reward -= 80.0
            truncated = True

        # UPDATE PREVIOUS DISTANCE
        self.prev_dist_to_goal = dist_to_goal
        self.prev_vel = self.vel.copy()

        # RENDER
        if self.render_mode == "human":
            self._render()

        # INFO
        info = {
            "dist_goal": float(dist_to_goal),
            "nearest_obstacle": float(nearest_dist),
            "success": is_success,
            "collision": is_collision,
            "episode_step": self.step_count,
            "reward": reward
        }
        if terminated or truncated:
            info["success"] = is_success
            info["collision"] = is_collision
            info["episode"] = {
                "r": reward,
                "l": self.step_count,
                "t": 0
            }

        return self._get_obs(), reward, terminated, truncated, info
    
    def _radar_scan(self):
        radar = np.ones(self.num_rays, dtype=np.float32)
        if np.linalg.norm(self.vel) > 1e-6:
            heading = np.arctan2(self.vel[1], self.vel[0])
        else:
            heading = 0.0
        angles = heading + np.linspace(-np.pi, np.pi, self.num_rays, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        for i, d in enumerate(directions):
            min_dist = self.radar_range
            for obs in self.obstacles:
                vec = obs - self.pos
                proj = np.dot(vec, d)
                if proj <= 0:
                    continue
                closest = self.pos + proj * d
                perp = np.linalg.norm(obs - closest)

                if perp >= self.obstacle_radius:
                    continue

                hit = proj - np.sqrt(max(self.obstacle_radius**2 - perp**2, 0.0))

                if hit > 0:
                   min_dist = min(min_dist, hit)

        # =========================
        # WALL INTERSECTION (SAFE)
        # =========================
            dx, dy = d

        # vertical walls
            if abs(dx) > 1e-6:
                t = (self.world_min - self.pos[0]) / dx
                if 0 < t < min_dist:
                    y = self.pos[1] + t * dy
                    if self.world_min <= y <= self.world_max:
                        min_dist = t

                t = (self.world_max - self.pos[0]) / dx
                if 0 < t < min_dist:
                    y = self.pos[1] + t * dy
                    if self.world_min <= y <= self.world_max:
                        min_dist = t

        # horizontal walls
            if abs(dy) > 1e-6:
                t = (self.world_min - self.pos[1]) / dy
                if 0 < t < min_dist:
                    x = self.pos[0] + t * dx
                    if self.world_min <= x <= self.world_max:
                        min_dist = t

                t = (self.world_max - self.pos[1]) / dy
                if 0 < t < min_dist:
                    x = self.pos[0] + t * dx
                    if self.world_min <= x <= self.world_max:
                        min_dist = t

            radar[i] = np.clip(min_dist / self.radar_range, 0.0, 1.0)
        return radar
        

    # OBSERVATION
    def _get_obs(self):
        # GOAL VECTOR
        goal_vec = self.goal - self.pos
        dist_goal = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (dist_goal + 1e-6)
        
        # HEADING
        speed = np.linalg.norm(self.vel)
        if speed > 1e-6:
            heading = self.vel / speed
        else:
            heading = np.zeros(2, dtype=np.float32)

        # WALL DISTANCES
        wall_left = self.pos[0] / self.world_size
        wall_right = (self.world_max - self.pos[0]) / self.world_size
        wall_bottom = self.pos[1] / self.world_size
        wall_top = (self.world_max - self.pos[1]) / self.world_size

        # RADAR
        radar = self._radar_scan()

        # OBSERVATION VECTOR
        obs = np.array([
            # goal direction (2)
            goal_dir[0], goal_dir[1],

            # heading direction (2)
            heading[0], heading[1],

            # normalized velocity (2)
            self.vel[0] / self.max_speed, self.vel[1] / self.max_speed,

            # normalized goal distance (1)
            dist_goal / self.world_size,

            # wall distances (4)
            wall_left, wall_right, wall_bottom, wall_top,

            # radar beams (24)
            *radar
        ], dtype=np.float32)

        # SAFETY CHECK
        assert len(obs) == self.observation_space.shape[0], (
            f"Observation size mismatch: "
            f"{len(obs)} != {self.observation_space.shape[0]}"
        )
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