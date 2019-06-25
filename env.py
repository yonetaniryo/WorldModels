import numpy as np
# import gym
import subprocess
import importlib


# To avoid unexpected bug
# def make_env(env_name, seed=-1, render_mode=False, model=None):
# 	if env_name == 'car_racing':
# 		from custom_envs.car_racing import CarRacing
# 		env = CarRacing()
# 		if (seed >= 0):
# 			env.seed(seed)
# 	elif env_name == 'car_racing_dream':
# 		from custom_envs.car_racing_dream import CarRacingDream
# 		env = CarRacingDream(model)
# 		if (seed >= 0):
# 			env.seed(seed)
# 	else:
# 		print("couldn't find this env")
# 
# 	return env

def make_env_car_racing(env_name, seed=-1, render_mode=False, model=None):
    if env_name == 'car_racing_11':
        from custom_envs.car_racing_11 import CustomEnv
    elif env_name == 'car_racing_12':
        from custom_envs.car_racing_12 import CustomEnv
    elif env_name == 'car_racing_13':
        from custom_envs.car_racing_13 import CustomEnv
    if env_name == 'car_racing_21':
        from custom_envs.car_racing_21 import CustomEnv
    elif env_name == 'car_racing_22':
        from custom_envs.car_racing_22 import CustomEnv
    elif env_name == 'car_racing_23':
        from custom_envs.car_racing_23 import CustomEnv
    if env_name == 'car_racing_31':
        from custom_envs.car_racing_31 import CustomEnv
    elif env_name == 'car_racing_32':
        from custom_envs.car_racing_32 import CustomEnv
    elif env_name == 'car_racing_33':
        from custom_envs.car_racing_33 import CustomEnv

    env = CustomEnv()
    if (seed >= 0):
        env.seed(seed)

    return env