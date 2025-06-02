import os
import sys

try:
    import time
    import gymnasium as gym
    import numpy as np
    import math
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000): # Set a concrete value for max_episode_steps
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', entry_point='openai_gym:OpenAIGymEnvironment', max_episode_steps=max_episode_steps)
        self.__timestep = int(self.getBasicTimeStep())

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-9, -9], dtype=np.float32),
            high=np.array([9, 9], dtype=np.float32)
        )

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
        # 5 proximity sensors + 2 ground sensors = 7 observations
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0,0,0,0,0],dtype=np.float32),
            high=np.array([1,1,1,1,1,1,1], dtype=np.float32))

        self.state = None

        print("Lista dos dispositivos do robot:")
        for i in range(self.getNumberOfDevices()):
            device = self.getDeviceByIndex(i)
            print(f"Device {i}: {device.getName()}")


        # Do all other required initializations
        self.thymio_node = self.getFromDef("ROBOT")
        if self.thymio_node is None:
            print("ERROR: 'ROBOT' node not found. Please check your .wbt file DEF name.")
            sys.exit(1)

        self.left_motor = self.getDevice("motor.left")
        self.right_motor = self.getDevice("motor.right")

        if self.left_motor is None or self.right_motor is None:
            print("ERROR: Motors 'motor.left' or 'motor.right' not found. Check device names.")
            sys.exit(1)

        self.left_motor.setPosition(float('inf'))   # Set motors to velocity mode
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0) # Ensure they start at 0 velocity
        self.right_motor.setVelocity(0.0)

        self.proximity_sensors = []
        for i in range(5):
            sensor = self.getDevice(f"prox.horizontal.{i}")
            if sensor is None:
                print(f"WARNING: Proximity sensor 'prox.horizontal.{i}' not found.")
                continue
            sensor.enable(self.__timestep)
            self.proximity_sensors.append(sensor)

        self.ground_sensors = []
        for i in range(2):
            sensor = self.getDevice(f"prox.ground.{i}")
            if sensor is None:
                print(f"WARNING: Ground sensor 'prox.ground.{i}' not found.")
                continue
            sensor.enable(self.__timestep)
            self.ground_sensors.append(sensor)

        self.__n = 0
        self.prev_pos = None
        self.visited_positions = []

        self.obstacles = []
        for i in range(5):  # or however many obstacles you have
            obstacle = self.getFromDef(f"OBSTACLE_{i+1}")
            if obstacle:
                self.obstacles.append(obstacle)

    def randomize_cubes(self):
        # List with the DEF names of the cubes
        cube_names = ["CUBE1", "CUBE2", "CUBE3", "CUBE4", "CUBE5", "CUBE6"]

        for cube_name in cube_names:
            cube_node = self.getFromDef(cube_name)
            if cube_node is None:
                print(f"Warning: cube {cube_name} not found")
                continue # Skip to the next cube if node is not found
            field = cube_node.getField("translation")
            if field is None:
                print(f"Warning: 'translation' field not found for cube {cube_name}")
                continue

            # Get current position to keep height (z) fixed at 1.1 and vary only x and y in a small range
            current_pos = field.getSFVec3f()  # [x, y, z]

            x = current_pos[0] + np.random.uniform(-0.2, 0.2)  # vary +- 0.2 on x-axis
            y = current_pos[1] + np.random.uniform(-0.2, 0.2)  # vary +- 0.2 on y-axis
            z = 1.1  # fix height

            # Update cube position
            field.setSFVec3f([x, y, z])


    def read_sensors(self):
        # Read values from proximity sensors
        proximity_values = {}
        for i, sensor in enumerate(self.proximity_sensors):
            val = sensor.getValue()
            proximity_values[f'prox.horizontal.{i}'] = val

        # Read values from ground sensors
        ground_values = {}
        for i, sensor in enumerate(self.ground_sensors):
            val = sensor.getValue()
            ground_values[f'prox.ground.{i}'] = val

        # Print sensor values (you can comment this out later if you want)
        # print("Proximity Sensors:")
        # for name, val in proximity_values.items():
        #     print(f"  {name}: {val:.2f}")

        # print("Ground Sensors:")
        # for name, val in ground_values.items():
        #     print(f"  {name}: {val:.2f}")

        # Normalize values to be between 0 and 1
        # Max values for Thymio II sensors: Proximity ~4000, Ground ~1000
        prox_norm = np.array(list(proximity_values.values())) / 4000.0
        ground_norm = np.array(list(ground_values.values())) / 1000.0

        # Concatenate into a single vector
        obs = np.concatenate((prox_norm, ground_norm))

        # print(f"Motor velocities (before reading sensors): Left: {self.left_motor.getVelocity():.2f}, Right: {self.right_motor.getVelocity():.2f}")

        return obs


    def apply_action(self, action):
        # Removed the multiplier. Speeds are now directly from the action, clipped to [-9, 9].

        print(f"ACTION apply_action: {action}")
        left_speed = float(np.clip(action[0], -9, 9))
        right_speed = float(np.clip(action[1], -9, 9))

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Debugging: Print the speeds that were *set*
        print(f"Applied speeds - Left: {left_speed:.2f}, Right: {right_speed:.2f}")
        # Debugging: Print the speeds *read back* from the motors immediately after setting
        print(f"Motor actual velocities (after set): Left: {self.left_motor.getVelocity():.2f}, Right: {self.right_motor.getVelocity():.2f}")


    def collision_detected(self):
        # Define collision if any frontal proximity sensor is very close
        # Sensors 0, 1, 2, 3, 4 correspond to front-left, front-center-left, front-center, front-center-right, front-right
        # Check central and slightly off-center front sensors
        proximity_vals = np.array([sensor.getValue() for sensor in self.proximity_sensors])
        # A value like 3500-4000 usually means very close or touching
        if np.any(proximity_vals[[1, 2, 3]] > 3500): # Check sensors 1, 2, 3 (front-center-left, front-center, front-center-right)
            return True
        return False


    def get_proximity_sensors(self):
        proximity_vals = np.array([sensor.getValue() for sensor in self.proximity_sensors])
        return np.clip(proximity_vals / 4000.0, 0, 1)

    def get_ground_sensors(self):
        ground_vals = np.array([sensor.getValue() for sensor in self.ground_sensors])
        return np.clip(ground_vals / 1000.0, 0, 1)


    def get_robot_position(self):
        # Get the robot's position in the environment
        trans_field = self.thymio_node.getField("translation")
        pos = trans_field.getSFVec3f()
        return np.array(pos[:2])  # x and y coordinates

    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)  # Step once to apply reset and get initial state

        self.__n = 0
        self.prev_pos = None
        self.visited_positions = []

        self.randomize_cubes()

        # ✅ GARANTIR MODO DE VELOCIDADE NOS MOTORES
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    # Allow some time for physics to settle after randomization
        for i in range(50):
            super().step(self.__timestep)

        init_state = self.read_sensors()
        return np.array(init_state).astype(np.float32), {}


    # Robot reward function
    # First, check if the robot is near a fall
    # Second, avoid collisions with objects
    # Third, reward exploration of new locations
    # Penalize already visited locations
    # Prefer positive linear velocities
    def compute_reward(self, action):
        reward = 0
        terminated = False

        proximity_sensors = self.get_proximity_sensors()
        ground_sensors = self.get_ground_sensors()

        # Penalty for being near an edge/fall (ground sensor values too low)
        if min(ground_sensors) < 0.2:  # Threshold for detecting edge
            reward -= 20
            terminated = True  # Episode terminates if it's about to fall

        # Penalty for being too close to an obstacle (high proximity sensor values)
        if max(proximity_sensors) > 0.8:  # Threshold for being very close
            reward -= 5

        # Strong penalty for collision
        if self.collision_detected():
            reward -= 5
            terminated = True  # Episode terminates on collision

        current_pos = self.get_robot_position()

        # Reward for movement (change in position)
        if self.prev_pos is not None:
            delta_pos = np.linalg.norm(current_pos - self.prev_pos)
            reward += 0.1 * delta_pos  # Small reward for moving

        self.prev_pos = current_pos

        # Initialize visited_positions if it doesn't exist
        if not hasattr(self, 'visited_positions'):
            self.visited_positions = []

        # Penalty for staying in already visited locations
        for pos in self.visited_positions:
            if np.linalg.norm(current_pos - pos) < 0.1:
                reward -= 0.2
                break
        self.visited_positions.append(current_pos)

        # Reward/penalty based on linear velocity
        vel_linear = (action[0] + action[1]) / 2
        if vel_linear > 0:
            reward += 0.5 * vel_linear
        else:
            reward -= 0.5 * abs(vel_linear)

        # ----------- Stagnation Check ------------
        STAGNATION_STEPS = 300  # Number of steps to consider
        MOVEMENT_THRESHOLD = 0.05  # Min total movement

        if len(self.visited_positions) >= STAGNATION_STEPS:
            recent_positions = self.visited_positions[-STAGNATION_STEPS:]
            total_movement = sum(
                np.linalg.norm(recent_positions[i] - recent_positions[i - 1])
                for i in range(1, len(recent_positions))
            )
            if total_movement < MOVEMENT_THRESHOLD:
                reward -= 10  # Optional penalty
                terminated = True
                print("Stagnation detected: episode terminated.")
    # ------------------------------------------

        return reward, terminated


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #
    def step(self, action):

        print(f"ACTION step: {action}")

        self.apply_action(action) # Apply the action (set motor velocities)

        # Advance the Webots simulation multiple basic time steps
        # This is where the robot physically moves in the simulation
        for i in range(20): # Advance 10 basic time steps per Gym step
            # IMPORTANT: The correct way to step the supervisor is self.step(self.__timestep)
            # You had a recursive call to self.step(self.__timestep) which would cause issues.
            # This line should call the parent Supervisor's step method.
            if Supervisor.step(self, self.__timestep) == -1: # Corrected call to parent step method
                print("Webots simulation ended prematurely during step loop.")
                return self.state.astype(np.float32), 0, True, False, {} # Return terminated if simulation ends
            # Debugging: Confirm that the Webots simulation is stepping
            # print(f"Webots simulation step {i+1} of 10.")

        self.state = self.read_sensors() # Read sensor values after simulation has advanced
        reward, terminated = self.compute_reward(action) # Compute reward and termination status
        print(f"Reward: {reward:.2f}, Terminated: {terminated}")

        return self.state.astype(np.float32), reward, terminated, False, {}


def train_ppo():
    env = OpenAIGymEnvironment()

    if not os.path.exists('./ppo_checkpoints'):
        os.makedirs('./ppo_checkpoints')

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=32,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./ppo_best_model/',
        log_path='./ppo_eval_logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("Starting PPO training...")
    model.learn(total_timesteps=500000, callback=[checkpoint_callback, eval_callback])
    print("PPO training finished.")

    model.save("ppo_thymio_final")
    print("Training concluded and model saved.")

def test_ppo_model():
    print("Starting PPO model test...")
    env = OpenAIGymEnvironment()
    try:
        model = PPO.load("ppo_thymio_final")  # Load the saved model
    except Exception as e:
        print(f"ERROR: Could not load model 'ppo_thymio_final'. Make sure it exists. Error: {e}")
        return

    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True) # Use deterministic=True for testing
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
        print(f"Test Step: {steps}, Action: [{action[0]:.2f}, {action[1]:.2f}], Reward: {reward:.2f}, Total Episode Reward: {episode_reward:.2f}")
        if steps >= env.spec.max_episode_steps:
            print(f"Episode reached max steps ({env.spec.max_episode_steps}). Terminating.")
            done = True
    print(f"Test episode finished. Total Reward: {episode_reward:.2f}, Total Steps: {steps}")
    print("PPO model test concluded.")


def main():
    # --- Current manual test loop (uncommented) ---
    # This part is for initial debugging of robot movement without PPO.
    # If the robot doesn't move here, the issue is not with PPO.
    """env = OpenAIGymEnvironment()

    obs, info = env.reset()  # initial reset

    # Ensure motors are in velocity mode and start at 0
    env.left_motor.setPosition(float('inf'))
    env.right_motor.setPosition(float('inf'))
    env.left_motor.setVelocity(0)
    env.right_motor.setVelocity(0)

    done = False
    steps = 0
    print("\n--- Starting manual movement test ---")
    while not done and steps < 100: # Run for 100 Gym steps
        action = np.array([5, 5]) # Hardcoded action to move forward
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Manual Test Step: {steps}, Action: [{action[0]:.2f}, {action[1]:.2f}], Reward: {reward:.2f}")
        done = terminated or truncated
        steps += 1
    print("--- Manual movement test finished ---\n")

    # --- Uncomment one of these lines to run training or testing with PPO ---
    """
    #train_ppo()
    test_ppo_model()

if __name__ == '__main__':
    main()