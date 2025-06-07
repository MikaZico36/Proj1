import os
import sys

from stable_baselines3.common.monitor import Monitor
from torch import nn

try:
    import time
    import gymnasium as gym
    import numpy as np
    import math
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


# Structure of a class to create an OpenAI Gym in Webots.
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000, enable_ground_reward=True, enable_collision_reward=True, enable_movement_reward=True, enable_linear_vel_reward=True, randomize_on_reset=True): # Set a concrete value for max_episode_steps
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', entry_point='openai_gym:OpenAIGymEnvironment', max_episode_steps=max_episode_steps)
        self.__timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-9, -9], dtype=np.float32),
            high=np.array([9, 9], dtype=np.float32)
        )

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
        # 5 proximity sensors + 2 ground sensors = 7 observations
        low_obs = np.array([0]*5 + [0]*2 + [-9, -9] + [-np.inf]*2 + [-np.pi] + [0], dtype=np.float32)
        high_obs = np.array([1]*5 + [1]*2 + [9, 9] + [np.inf]*2 + [np.pi] + [1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)


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

        self.boxes = []
        for i in range(1, 6):
            box_node = self.getFromDef(f"BOX{i}")
            if box_node:
                self.boxes.append(box_node)
            else:
                print(f"WARNING: BOX{i} not found in the scene.")

        self.curriculum_stage = 0
        self.obstacles = []
        for i in range(5):
            obstacle = self.getFromDef(f"OBSTACLE_{i+1}")
            if obstacle:
                self.obstacles.append(obstacle)
        self.episode_number = 0
        self.enable_ground_reward = enable_ground_reward
        self.enable_collision_reward = enable_collision_reward
        self.enable_movement_reward = enable_movement_reward
        self.enable_linear_vel_reward = enable_linear_vel_reward
        self.enable_randomize_on_reset = randomize_on_reset

        self.episode_reward = 0.0
        self.episode_number = 0





    #Deteta a colisão do Thymio com os obstáculos
    def collision_detected(self):
        proximity_vals = np.array([sensor.getValue() for sensor in self.proximity_sensors])
        if np.any(proximity_vals[[1, 2, 3]] > 3000):
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

    #Coloca os obstáculos de forma aleatória no cenário.
    def randomize_cubes(self):
        cube_names = ["CUBE1", "CUBE2", "CUBE3", "CUBE4", "CUBE5", "CUBE6"]

        for cube_name in cube_names:
            cube_node = self.getFromDef(cube_name)
            if cube_node is None:
                continue

            pos_field = cube_node.getField("translation")
            if pos_field is not None:
                current_pos = pos_field.getSFVec3f()
                x = current_pos[0] + np.random.uniform(-0.2, 0.2)
                y = current_pos[1] + np.random.uniform(-0.2, 0.2)
                z = 1
                pos_field.setSFVec3f([x, y, z])

            rot_field = cube_node.getField("rotation")
            if rot_field is not None:
                angle = np.random.uniform(0, 2 * np.pi)
                rot_field.setSFRotation([0, 0, 1, angle])

            size_field = cube_node.getField("size")
            if size_field is not None:
                new_size = [
                    np.random.uniform(0.1, 0.2),
                    np.random.uniform(0.1, 0.2),
                    np.random.uniform(0.1, 0.2)
                ]
                size_field.setSFVec3f(new_size)



    def check_box_visits(self):
        current_pos = self.get_robot_position()
        visit_radius = 0.3  # raio para considerar visita (ajuste conforme necessário)

        rewards = 0
        for i, box in enumerate(self.boxes, start=1):
            box_pos_field = box.getField("translation")
            box_pos = np.array(box_pos_field.getSFVec3f()[:2])

            dist = np.linalg.norm(current_pos - box_pos)
            if dist < visit_radius and (i not in self.visited_boxes):
                self.visited_boxes.add(i)
                rewards += 2
                #print(f"Visited BOX{i}!")

        if len(self.visited_boxes) == len(self.boxes):
            #print("All boxes visited! Resetting visits and giving bonus reward.")
            self.visited_boxes = set()
            rewards += 5

        return rewards


    #Função que lê os sensores do Thymio
    # Lê sensores + estado interno do robô
    def read_sensors(self):

        prox_norm = self.get_proximity_sensors()
        ground_norm = self.get_ground_sensors()

        vel_left = self.left_motor.getVelocity()
        vel_right = self.right_motor.getVelocity()

        pos = self.get_robot_position()

        rot = self.thymio_node.getField("rotation").getSFRotation()
        angle = rot[3]

        time_step_norm = self.steps_since_reset / self.max_episode_steps

        obs = np.concatenate([
            prox_norm,
            ground_norm,
            [vel_left, vel_right],
            pos,
            [angle],
            [time_step_norm]
        ])
        return obs.astype(np.float32)

    #Recebe a ação a tomar, interpreta-a a executa-a
    def apply_action(self, action, training=True):
        if training:
            noise = np.random.normal(0, 0.2, size=action.shape)
            action = np.clip(action + noise, -9, 9)

        left_speed = float(np.clip(action[0], -9, 9))
        right_speed = float(np.clip(action[1], -9, 9))

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)


    # Recria o cenário de treino do Thymio
    def reset(self, seed=None, options=None):
        self.episode_number += 1
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        self.__n = 0
        self.prev_pos = None
        self.visited_positions = []
        self.steps_since_reset = 0
        self.visited_boxes = set()
        self.episode_reward = 0.0


        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        if self.enable_randomize_on_reset:
            self.randomize_cubes()

        rotation_field = self.thymio_node.getField("rotation")
        if rotation_field is not None:
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_field.setSFRotation([0, 0, 1, angle])

        for i in range(20):
            super().step(self.__timestep)

        init_state = self.read_sensors()
        return np.array(init_state).astype(np.float32), {}



    def reward(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Pega sensores
        prox = self.get_proximity_sensors()
        ground = self.get_ground_sensors()
        position = self.get_robot_position()

        # Estágio 0: só recompensa por andar (velocidade linear positiva)
        if self.curriculum_stage >= 0:
            speed = (action[0] + action[1]) / 2
            if speed > 0:
                reward += 1.0
            else:
                reward -= 0.5

        # Estágio 1: recompensa por evitar obstáculos
        if self.curriculum_stage >= 1:
            front_prox = prox[1:4]
            if np.any(front_prox > 0.3):
                reward -= np.mean(front_prox) * 2  # penaliza proximidade de obstáculos

        # Estágio 2: evitar quedas
        if self.curriculum_stage >= 2:
            if np.any(ground < 0.5):
                reward -= 10
                terminated = True

        # Estágio 3: incentivo à exploração
        if self.curriculum_stage >= 3:
            # Usa a distância da posição anterior para incentivar movimento
            if self.prev_pos is not None:
                dist = np.linalg.norm(position - self.prev_pos)
                reward += dist * 2
            self.prev_pos = position


        if self.curriculum_stage >= 4:
            for box in self.boxes:
                pos_field = box.getField("translation")
                box_position = np.array(pos_field.getSFVec3f())
                box_distance = np.linalg.norm(np.array(position[:2]) - np.array(box_position[:2]))
                if box_distance < 0.1 and box not in self.visited_boxes:
                    self.visited_boxes.add(box)
                    reward += 5

        self.steps_since_reset += 1
        if self.steps_since_reset >= self.max_episode_steps:
            truncated = True

        # Você pode ajustar truncamento por steps etc.

        return reward, terminated, truncated


    def step(self, action, training=True):
        self.apply_action(action, training=training)

        if Supervisor.step(self, self.__timestep) == -1:
            return self.state.astype(np.float32), 0, True, False, {}

        self.state = self.read_sensors()
        reward, terminated, truncated = self.reward(action)
        self.steps_since_reset += 1

        self.episode_reward += reward  # acumula

        if terminated or truncated:
            print(f"Episode {self.episode_number} reward: {self.episode_reward:.2f} stage: {self.curriculum_stage}")

        return self.state.astype(np.float32), reward, terminated, truncated, {}



def heuristic_policy(env):
    prox = env.get_proximity_sensors()
    ground = env.get_ground_sensors()
    front_sensors = prox[1:4]
    threshold_obstacle = 0.3
    threshold_ground = 0.5

    # Evitar quedas: se algum sensor do chão detectar borda, recua
    if np.any(ground < threshold_ground):
        left_speed = -3.0
        right_speed = -3.0
        return np.array([left_speed, right_speed], dtype=np.float32)

    # Evitar obstáculos: se sensor frontal detecta obstáculo, vira aleatório
    if np.any(front_sensors > threshold_obstacle):
        turn_dir = np.random.choice([-1, 1])
        left_speed = -5.0 * turn_dir
        right_speed = 5.0 * turn_dir
        return np.array([left_speed, right_speed], dtype=np.float32)

    # Caso contrário, anda reto com velocidade positiva
    left_speed = 6.0
    right_speed = 6.0
    return np.array([left_speed, right_speed], dtype=np.float32)



class CurriculumCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.recent_rewards = []
        self.window_size = 20
        self.min_variation = 0.1  # limiar para detectar estagnação da recompensa

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards")
        if reward is None:
            return True

        if isinstance(reward, (list, np.ndarray)):
            reward = reward[-1]

        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)

        if len(self.recent_rewards) == self.window_size:
            variation = np.std(self.recent_rewards)
            current_stage = self.env.curriculum_stage

            if variation <= self.min_variation and current_stage < 4:
                new_stage = current_stage + 1
                self.env.curriculum_stage = new_stage
                if self.verbose > 0:
                    print(f"Curriculum stage updated to {new_stage} at timestep {self.num_timesteps}")
                self.recent_rewards.clear()  # reseta histórico após subir estágio para evitar múltiplos triggers

        return True

def train_recurrent_ppo(env, total_timesteps=200_000, stage_length=50_000, model_save_path="./recurret_test_ppo_thymio"):

    # Cria diretório se não existir
    os.makedirs(model_save_path, exist_ok=True)

    # Monitor para logging
    env = Monitor(env)

    # Policy personalizada simples (opcional), para RNN
    policy_kwargs = dict(
        net_arch = dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh
    )

    # Inicializa RecurrentPPO
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        device="auto",
        seed=42
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=model_save_path, name_prefix="rppo_thymio")
    curriculum_callback = CurriculumCallback(env.env, verbose=1)

    # Treina o modelo
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, curriculum_callback])

    # Salva modelo final
    model.save(os.path.join(model_save_path, "rppo_thymio_final"))

    print("Treinamento concluído!")

    return model



if __name__ == "__main__":
    env = OpenAIGymEnvironment(max_episode_steps=1000)
    trained_model = train_recurrent_ppo(env)