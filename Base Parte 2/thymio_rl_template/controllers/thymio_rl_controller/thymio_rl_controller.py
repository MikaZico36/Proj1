#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

import sys
import os
try:

    print("Python executado pelo Webots:")
    print("sys.executable:", sys.executable)
    print("sys.version:", sys.version)
    print("PATH:", os.environ.get("PATH"))
    import time
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor
    from stable_baselines3.common.vec_env import DummyVecEnv
    #from sb3_contrib.common.recurrent.vec_env import VecTransposeWrapper


except ImportError as e:
    print(f"Erro ao importar dependências: {e}")
    exit()


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.__timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps

        # Referências do robô
        self.robot_node = self.getFromDef("ROBOT")
        if self.robot_node is None:
            raise RuntimeError("Nó ROBOT não encontrado. Verifique o DEF no Webots.")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        # Motores
        self.left_motor = self.getDevice("motor.left")
        self.right_motor = self.getDevice("motor.right")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensores IR horizontais e de chão
        self.ir_sensors = [
            self.getDevice(f'prox.horizontal.{i}') for i in [0, 2, 4]
        ] + [
            self.getDevice('prox.ground.0'),
            self.getDevice('prox.ground.1')
        ]
        for sensor in self.ir_sensors:
            sensor.enable(self.__timestep)

        # Espaço de ação e observação
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        self.state = None
        self.step_count = 0
        self.max_speed = 9.53

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset da simulação
        self.simulationReset()
        self.simulationResetPhysics()
        self.translation_field.setSFVec3f([0.0, 0.0, 0.0])  # Posição inicial
        self.rotation_field.setSFRotation([0, 1, 0, 0])     # Orientação inicial
        for _ in range(15):
            super().step(self.__timestep)

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.step_count = 0
        self.state = self._get_observation()

        return self.state, {}

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
        reward = 0.0

        forward_motion = max(0.0, (action[0] + action[1]) / 2.0)
        reward += forward_motion * 2.0

        obstacle_penalty = np.max(state[:3]) * 1.5
        reward -= obstacle_penalty

        ground_penalty = (1.0 - min(state[3], state[4])) * 5.0
        reward -= ground_penalty

        return reward

    def _check_termination(self, state):
        collision = np.max(state[:3]) > 0.9
        hole = min(state[3], state[4]) < 0.2
        return bool(collision or hole)

    def render(self):
        pass  # Não necessário para Webots


def main():
    # Criação de ambiente vetorizado com compatibilidade para RecurrentPPO
    def make_env():
        return OpenAIGymEnvironment()

    vec_env = DummyVecEnv([make_env])
    #vec_env = VecTransposeWrapper(vec_env)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        device="auto"
    )

    callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints',
        name_prefix='thymio_model'
    )

    print("Iniciando treinamento...")
    model.learn(total_timesteps=100000, callback=callback)
    model.save("thymio_recurrent_ppo")

    # Teste
    print("Testando modelo treinado...")
    obs, _ = vec_env.reset()
    lstm_states = None
    episode_reward = 0
    dones = False

    while not dones:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, terminated, truncated, _ = vec_env.step(action)
        dones = terminated or truncated
        episode_reward += reward[0]
        time.sleep(0.05)

    print(f"Recompensa total do episódio: {episode_reward}")


if __name__ == '__main__':
    main()

"""try:
    import time
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

except ImportError:
    #sys.exit('Please make sure you have all dependencies installed.')
    print("exit")
    exit()

class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.__timestep = int(self.getBasicTimeStep())
        
        # self.evaluation_start_time = 0
        self.robot_node = self.getFromDef("ROBOT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        # Motores
        self.left_motor = self.getDevice("motor.left")
        self.right_motor = self.getDevice("motor.right")
        
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensores
        self.__ir_0 = self.getDevice('prox.horizontal.0')
        self.__ir_1 = self.getDevice('prox.horizontal.1')
        self.__ir_2 = self.getDevice('prox.horizontal.2')
        self.__ir_3 = self.getDevice('prox.horizontal.3')
        self.__ir_4 = self.getDevice('prox.horizontal.4')
        self.__ir_5 = self.getDevice('prox.horizontal.5')
        self.__ir_6 = self.getDevice('prox.horizontal.6')
        self.__ir_7 = self.getDevice('prox.ground.0')
        self.__ir_8 = self.getDevice('prox.ground.1')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.__ir_0.enable(self.__timestep)
        self.__ir_1.enable(self.__timestep)
        self.__ir_2.enable(self.__timestep)
        self.__ir_3.enable(self.__timestep)
        self.__ir_4.enable(self.__timestep)
        self.__ir_5.enable(self.__timestep)
        self.__ir_6.enable(self.__timestep)
        self.__ir_7.enable(self.__timestep)
        self.__ir_8.enable(self.__timestep)

        # Espaço de ação: velocidades normalizadas [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observação: 5 prox frontais + 2 sensores de chão, normalizados [0, 1]
        self.observation_space = gym.spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.__n = 0
        self.max_speed = 9.53  # valor de escala das ações

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        # Inicializar motores
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Reset interno
        self.__n = 0

        # Estabilizar física
        for _ in range(15):
            super().step(self.__timestep)

        # Obter estado inicial
        self.state = self._get_observation()
        return self.state, {}

    def _get_observation(self):
        # Normalizar leitura dos sensores
        self.sensors = [self.__ir_0,self.__ir_2,self.__ir_4]
        self.ground_sensors = [self.getDevice(f'prox.ground.{i}').getValue() / 1000.0 for i in range(2)] # valor esperado ~0–1000
        return np.array(self.sensors + self.ground_sensors, dtype=np.float32)

    def step(self, action):
        self.__n += 1

        # Aplicar velocidades escaladas
        left_speed = float(action[0]) * self.max_speed
        right_speed = float(action[1]) * self.max_speed
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Avançar simulação
        for _ in range(10):
            super().step(self.__timestep)

        # Observar novo estado
        self.state = self._get_observation()

        # Recompensa: evitar colisões (prox baixos), manter velocidade positiva
        reward = 0.0
        reward += 0.5 * (action[0] + action[1])  # velocidade para frente
        reward -= 2.0 * np.max(self.state[:5])  # penalizar proximidade a obstáculos
        reward -= 5.0 * (1.0 - min(self.state[5], self.state[6]))  # penalizar buraco

        # Terminação por colisão ou queda
        terminated = bool(np.max(self.state[:5]) > 0.9 or min(self.state[5], self.state[6]) < 0.2)
        truncated = self.__n >= 500

        return self.state, reward, terminated, truncated, {}

def main():
    
    env = OpenAIGymEnvironment()

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints',
        name_prefix='thymio_model'
    )

    model.learn(total_timesteps=100000, callback=callback)
    model.save("thymio_recurrent_ppo")

    # Testar o modelo treinado
    obs, _ = env.reset()
    lstm_states = None
    dones = False

    while not dones:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        dones = terminated or truncated
        time.sleep(0.05)

if __name__ == '__main__':
    main()"""

""" try:
    import time
    import gymnasium as gym
    import numpy as np
    import math
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

except ImportError:
    #sys.exit('Please make sure you have all dependencies installed.')
    print("exit")
    exit()

#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps = 1000):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', entry_point='openai_gym:OpenAIGymEnvironment', max_episode_steps=max_episode_steps)
        self.__timestep = int(self.getBasicTimeStep())

        self.left_motor = self.getDevice("motor.left")
        self.right_motor = self.getDevice("motor.right")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Sensores de proximidade
        self.prox_sensors = [self.getDevice(f"prox.horizontal.{i}") for i in range(7)]
        for sensor in self.prox_sensors:
            sensor.enable(self.__timestep)

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-10.0, -10.0]), 
            high=np.array([10.0, 10.0]), dtype=np.float32)

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
         self.observation_space = gym.spaces.Box(
            low=np.array([7, 7]), 
            high=np.array([7, 4096.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.full(7, 4096.0, dtype=np.float32),
            dtype=np.float32
)

        self.state = None
        
        
        # Do all other required initializations
        self.__n = 0


    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        # initialize the sensors, reset the actuators, randomize the environment
        # See how in Lab 1 code
        # Parar motores
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # you may need to iterate a few times to let physics stabilize
        for i in range(15):
            super().step(self.__timestep)

        # set the initial state vector to return
        init_state = [sensor.getValue() for sensor in self.prox_sensors]
        self.state = np.array(init_state)

        self.__n = 0
        # return np.array(init_state).astype(np.float32), {}
        return np.array(init_state, dtype=np.float32), {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):

        self.__n = self.__n + 1

        # start by applying the action in the robot actuators
        # See how in Lab 1 code
        left_speed, right_speed = action
        self.left_motor.setVelocity(float(left_speed))
        self.right_motor.setVelocity(float(right_speed))

        # let the action to effect for a few timesteps
        for i in range(10):
            super().step(self.__timestep)


        # set the state that resulted from applying the action (consulting the robot sensors)
        self.state = np.array([sensor.getValue() for sensor in self.prox_sensors])

        # compute the reward that results from applying the action in the current state
        # Calcular recompensa (exemplo: evitar obstáculos)
        min_dist = np.min(self.state)
        reward = 1.0 - (min_dist / 4096.0)

        # set termination and truncation flags (bools)
        # Termina o episódio se houver colisão (exemplo: sensor frontal muito ativo)
        terminated = bool(self.state[3] > 3500)
        truncated = self.__n >= self.spec.max_episode_steps

        # return self.state.astype(np.float32), reward, terminated, truncated, {}
        return self.state.astype(np.float32), reward, terminated, truncated, {}


def main():

    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()

    # Treinamento com RecurrentPPO
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='thymio_model')

    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    model.save("thymio_recurrent_ppo")

    # Teste do modelo treinado
    obs, _ = env.reset()
    lstm_states = None
    dones = False

    while not dones:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        dones = terminated or truncated
        time.sleep(0.05)

    # Code to train and save a model
    # For the PPO case, see how in Lab 7 code
    # For the RecurrentPPO case, consult its documentation
    # ...

    # Code to load a model and run it
    # For the RecurrentPPO case, consult its documentation
    # ...


if __name__ == '__main__':
    main() """
