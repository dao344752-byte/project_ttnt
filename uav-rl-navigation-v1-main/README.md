# UAV 2D RL Navigation

This project trains a UAV agent using PPO to reach a goal while avoiding an obstacle.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

######################################################
python3 train_ppo.py
######################################################
python3 demo.py
######################################################
