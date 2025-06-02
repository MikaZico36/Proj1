import math
import time
from random import uniform

try:
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from controller import Supervisor
    from stable_baselines3.common.vec_env import DummyVecEnv
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Erro ao importar dependências: {e}")
    exit()


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=500, randomize_boxes=True):
        super().__init__()
        self.__timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps
        self.randomize_boxes = randomize_boxes

        self.robot_node = self.getFromDef("ROBOT")
        if self.robot_node is None:
            raise RuntimeError("Nó ROBOT não encontrado. Verifique o DEF no Webots.")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.left_motor = self.getDevice("motor.left")
        self.right_motor = self.getDevice("motor.right")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.ir_sensors = [self.getDevice(f'prox.horizontal.{i}') for i in [0, 2, 4]] + \
                          [self.getDevice('prox.ground.0'), self.getDevice('prox.ground.1')]
        for sensor in self.ir_sensors:
            sensor.enable(self.__timestep)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        self.state = None
        self.step_count = 0
        self.max_speed = 9.53
        self.box_positions = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()

        self.translation_field.setSFVec3f([0.0, 0.0, 0.0])
        random_angle = uniform(0, 2 * math.pi)
        self.rotation_field.setSFRotation([0, 1, 0, random_angle])

        self.box_positions.clear()
        n_boxes = 5
        min_distance = 0.3

        for i in range(1, n_boxes + 1):
            box_node = self.getFromDef(f"BOX{i}")
            if box_node is None:
                print(f"Aviso: Nó BOX{i} não encontrado, ignorando.")
                continue
            box_translation = box_node.getField("translation")
            box_rotation = box_node.getField("rotation")

            if self.randomize_boxes:
                pos = self._generate_non_overlapping_position(min_distance)
            else:
                # Posições fixas para ambiente constante
                fixed_positions = [
                    (0.5, 0.5),
                    (-0.5, 0.5),
                    (0.5, -0.5),
                    (-0.5, -0.5),
                    (0.0, 0.6)
                ]
                pos = fixed_positions[i - 1]

            box_translation.setSFVec3f([pos[0], 0.025, pos[1]])
            angle = uniform(0, 2 * math.pi)
            box_rotation.setSFRotation([0, 1, 0, angle])

            self.box_positions.append(pos)

        for _ in range(15):
            super().step(self.__timestep)

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.step_count = 0
        self.state = self._get_observation()

        return self.state, {}

    def _generate_non_overlapping_position(self, min_dist):
        while True:
            x = uniform(-0.6, 0.6)
            z = uniform(-0.6, 0.6)
            if abs(x) < 0.2 and abs(z) < 0.2:
                continue
            too_close = False
            for (px, pz) in self.box_positions:
                dist = math.sqrt((x - px) ** 2 + (z - pz) ** 2)
                if dist < min_dist:
                    too_close = True
                    break
            if not too_close:
                return (x, z)

    def _get_observation(self):
        readings = [s.getValue() / 1000.0 for s in self.ir_sensors]
        return np.array(readings, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        left_speed = float(np.clip(action[0], -1, 1)) * self.max_speed
        right_speed = float(np.clip(action[1], -1, 1)) * self.max_speed
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        for _ in range(10):
            super().step(self.__timestep)

        new_state = self._get_observation()
        self.state = new_state
        reward = self._calculate_reward(action, new_state)
        terminated = self._check_termination(new_state)
        truncated = self.step_count >= self.max_episode_steps
        return new_state, reward, terminated, truncated, {}

    def _calculate_reward(self, action, state):
        forward_motion = max(0.0, (action[0] + action[1]) / 2.0)
        obstacle_penalty = np.mean(state[:3])
        ground_penalty = 1.0 - min(state[3], state[4])
        reward = 2.0 * forward_motion - 1.0 * obstacle_penalty - 3.0 * ground_penalty
        return reward

    def _check_termination(self, state):
        collision = np.max(state[:3]) > 0.9
        hole = min(state[3], state[4]) < 0.2
        return bool(collision or hole)

    def render(self):
        pass


def main():
    def make_env():
        # Escolhe True para aleatório, False para ambiente constante
        return OpenAIGymEnvironment(randomize_boxes=True)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_ppo_mlp/",
        device="auto"
    )

    callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints_ppo_mlp',
        name_prefix='thymio_ppo_mlp'
    )

    print("Iniciando treinamento PPO MLP (não recorrente)...")

    model.learn(total_timesteps=100000, callback=callback)
    model.save("thymio_ppo_mlp")

    print("Treinamento concluído! Testando modelo...")

    obs, _ = vec_env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = vec_env.step(action)
        done = terminated or truncated
        episode_reward += reward[0]
        time.sleep(0.05)

    print(f"Recompensa total do episódio de teste: {episode_reward}")

    # Se quiser plotar recompensas, aqui você pode implementar coleta no callback
    # ou treinar com Monitor para registrar recompensas


if __name__ == '__main__':
    main()
