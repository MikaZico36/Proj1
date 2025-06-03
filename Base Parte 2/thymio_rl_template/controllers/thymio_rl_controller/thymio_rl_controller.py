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

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
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
        for i in range(5):
            obstacle = self.getFromDef(f"OBSTACLE_{i+1}")
            if obstacle:
                self.obstacles.append(obstacle)

#Coloca os obstáculos de forma aleatória no cenário.
    def randomize_cubes(self):
        cube_names = ["CUBE1", "CUBE2", "CUBE3", "CUBE4", "CUBE5", "CUBE6"]

        for cube_name in cube_names:
            cube_node = self.getFromDef(cube_name)
            if cube_node is None:
                continue

            # Aleatoriza posição (mantém altura Z fixa)
            pos_field = cube_node.getField("translation")
            if pos_field is not None:
                current_pos = pos_field.getSFVec3f()
                x = current_pos[0] + np.random.uniform(-0.2, 0.2)
                y = current_pos[1] + np.random.uniform(-0.2, 0.2)
                z = 1.1  # manter altura
                pos_field.setSFVec3f([x, y, z])

            # Aleatoriza rotação em torno do eixo Z (Y em Webots)
            rot_field = cube_node.getField("rotation")
            if rot_field is not None:
                angle = np.random.uniform(0, 2 * np.pi)
                rot_field.setSFRotation([0, 1, 0, angle])

            # Aleatoriza tamanho (máximo: 0.2 como no .wbt)
            size_field = cube_node.getField("size")
            if size_field is not None:
                new_size = [
                    np.random.uniform(0.1, 0.2),  # Largura
                    np.random.uniform(0.1, 0.2),  # Profundidade
                    np.random.uniform(0.1, 0.2)   # Altura
                ]
                size_field.setSFVec3f(new_size)

    #Função que lê os sensores do Thymio
    def read_sensors(self):
        proximity_values = {}
        for i, sensor in enumerate(self.proximity_sensors):
            val = sensor.getValue()
            proximity_values[f'prox.horizontal.{i}'] = val

        ground_values = {}
        for i, sensor in enumerate(self.ground_sensors):
            val = sensor.getValue()
            ground_values[f'prox.ground.{i}'] = val


        prox_norm = np.array(list(proximity_values.values())) / 4000.0
        ground_norm = np.array(list(ground_values.values())) / 1000.0

        obs = np.concatenate((prox_norm, ground_norm))

        return obs

#Recebe a ação a tomar, interpreta-a a executa-a
    def apply_action(self, action):

        left_speed = float(np.clip(action[0], -9, 9))
        right_speed = float(np.clip(action[1], -9, 9))

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    #Deteta a colisão do Thymio com os obstáculos
    def collision_detected(self):
        proximity_vals = np.array([sensor.getValue() for sensor in self.proximity_sensors])
        if np.any(proximity_vals[[1, 2, 3]] > 3500):
            return True
        return False

    #Retorna os valores dos sensores de proximidade
    def get_proximity_sensors(self):
        proximity_vals = np.array([sensor.getValue() for sensor in self.proximity_sensors])
        return np.clip(proximity_vals / 4000.0, 0, 1)

    #Retorna os valores dos sensores de proximidade
    def get_ground_sensors(self):
        ground_vals = np.array([sensor.getValue() for sensor in self.ground_sensors])
        return np.clip(ground_vals / 1000.0, 0, 1)


    #Retorna a posição do Thymio
    def get_robot_position(self):
        trans_field = self.thymio_node.getField("translation")
        pos = trans_field.getSFVec3f()
        return np.array(pos[:2])

    #
    # Recria o cenário de treino do Thymio
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        self.__n = 0
        self.prev_pos = None
        self.visited_positions = []

        self.randomize_cubes()

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        rotation_field = self.thymio_node.getField("rotation")
        if rotation_field is not None:
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_field.setSFRotation([0, 0, 1, angle])

        for i in range(20):
            super().step(self.__timestep)

        init_state = self.read_sensors()
        return np.array(init_state).astype(np.float32), {}


#Função de recompensa do Thymio
    def compute_reward(self, action):
        reward = 0
        terminated = False

        proximity_sensors = self.get_proximity_sensors()
        ground_sensors = self.get_ground_sensors()

        if min(ground_sensors) < 0.05:
            reward -= 20
            terminated = True

        if max(proximity_sensors) > 0.9:
            reward -= 5

        if self.collision_detected():
            reward -= 5
            terminated = True

        current_pos = self.get_robot_position()

        if self.prev_pos is not None:
            delta_pos = np.linalg.norm(current_pos - self.prev_pos)
            reward += 0.1 * delta_pos

        self.prev_pos = current_pos

        if not hasattr(self, 'visited_positions'):
            self.visited_positions = []

        for pos in self.visited_positions:
            if np.linalg.norm(current_pos - pos) < 0.1:
                reward -= 0.2
                break
        self.visited_positions.append(current_pos)

        vel_linear = (action[0] + action[1]) / 2
        if vel_linear > 0:
            reward += 0.5 * vel_linear
        else:
            reward -= 0.5 * abs(vel_linear)

        STAGNATION_STEPS = 100
        MOVEMENT_THRESHOLD = 0.05

        if len(self.visited_positions) >= STAGNATION_STEPS:
            recent_positions = self.visited_positions[-STAGNATION_STEPS:]
            total_movement = sum(
                np.linalg.norm(recent_positions[i] - recent_positions[i - 1])
                for i in range(1, len(recent_positions))
            )
            if total_movement < MOVEMENT_THRESHOLD:
                reward -= 10
                terminated = True

        return reward, terminated


    #
    # Corre um Timestamp do teste
    #
    def step(self, action):


        self.apply_action(action)

        for i in range(10):
            if Supervisor.step(self, self.__timestep) == -1:
                return self.state.astype(np.float32), 0, True, False, {}

        self.state = self.read_sensors()
        reward, terminated = self.compute_reward(action)
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
    train_ppo()
    #test_ppo_model()

if __name__ == '__main__':
    main()