import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
def draw_drone(ax, pos, vel, scale=1.0):
    speed = np.linalg.norm(vel)
    if speed > 1e-6:
        heading = vel / speed
    else:
        heading = np.array([1.0, 0.0])
    side = np.array([-heading[1], heading[0]])
    # SIZE
    arm_len = 0.20 * scale
    body_radius = 0.1 * scale
    rotor_radius = 0.070 * scale
    arrow_len = 0.50 * scale
    # ROTOR POSITIONS
    p1 = pos + heading * arm_len + side * arm_len
    p2 = pos + heading * arm_len - side * arm_len
    p3 = pos - heading * arm_len + side * arm_len
    p4 = pos - heading * arm_len - side * arm_len
    # X FRAME
    ax.plot([p1[0], p4[0]], [p1[1], p4[1]], color="black", linewidth=2.2, solid_capstyle="round", zorder=10)
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color="black", linewidth=2.2, solid_capstyle="round", zorder=10)
    # ROTORS
    for r in [p1, p2, p3, p4]:
        outer = Circle(r, rotor_radius, fill=False, edgecolor="black", linewidth=1.4, zorder=12)
        ax.add_patch(outer)
        hub = Circle(r, rotor_radius * 0.18, color="black", zorder=13)
        ax.add_patch(hub)
        blade = rotor_radius * 0.7
        ax.plot([r[0] - blade, r[0] + blade], [r[1], r[1]], color="gray", linewidth=0.8, alpha=0.7, zorder=11)
        ax.plot([r[0], r[0]], [r[1] - blade, r[1] + blade], color="gray", linewidth=0.8, alpha=0.7, zorder=11)
    # BODY
    body = Circle(pos, body_radius, facecolor="dodgerblue", edgecolor="black", linewidth=0.8, zorder=15)
    ax.add_patch(body)
    # NOSE
    nose = pos + heading * (body_radius * 2.0)
    ax.plot([pos[0], nose[0]], [pos[1], nose[1]], color="dodgerblue", linewidth=2.5, zorder=16)
    # VELOCITY ARROW
    arrow = FancyArrowPatch(pos, pos + heading * arrow_len, arrowstyle="-|>", mutation_scale=9, linewidth=1.8, color="#4d88ff", alpha=0.95, zorder=8)
    ax.add_patch(arrow)

def draw_goal(ax, goal):
    ax.scatter(goal[0], goal[1], marker="*", s=500, c="white", edgecolors="green", linewidths=2.5, zorder=30)
    ax.text(goal[0], goal[1] + 0.35, "GOAL", color="green", fontsize=10, fontweight="bold", ha="center", zorder=31)

def draw_obstacle(ax, center, radius):
    obs = Circle(center, radius, color="red", alpha=0.6, zorder=2)
    ax.add_patch(obs)

def draw_danger_zone(ax, center, radius):
    zone = Circle(center, radius, fill=False, linestyle="--", linewidth=1.2, color="red", alpha=0.35, zorder=1)
    ax.add_patch(zone)
