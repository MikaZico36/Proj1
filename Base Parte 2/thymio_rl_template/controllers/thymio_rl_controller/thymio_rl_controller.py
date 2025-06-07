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
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


# Structure of a class to create an OpenAI Gym in Webots.
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000, enable_ground_reward=True, enable_collision_reward=True, enable_movement_reward=True, enable_linear_vel_reward=True, randomize_on_reset=True):
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
        # 7 sensores = 5 sensores de proximidade e 2 sensores de chão
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

        self.boxes = []
        for i in range(1, 6):
            box_node = self.getFromDef(f"BOX{i}")
            if box_node:
                self.boxes.append(box_node)
            else:
                print(f"WARNING: BOX{i} not found in the scene.")


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


    #Verifica se o robô está a visitar uma nova plataforma
    def check_box_visits(self):
        current_pos = self.get_robot_position()
        visit_radius = 0.3

        rewards = 0
        for i, box in enumerate(self.boxes, start=1):
            box_pos_field = box.getField("translation")
            box_pos = np.array(box_pos_field.getSFVec3f()[:2])

            dist = np.linalg.norm(current_pos - box_pos)
            if i != 1 or len(self.visited_boxes) > 1:

                if dist < visit_radius and (i not in self.visited_boxes):
                    self.visited_boxes.add(i)
                    rewards += 5 + self.steps_since_reset * 0.1
                    print(f"Visited BOX{i}!")

        if len(self.visited_boxes) == len(self.boxes):
            print("All boxes visited! Resetting visits and giving bonus reward.")
            self.visited_boxes = set()
            rewards += 10

        return rewards


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
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        self.__n = 0
        self.prev_pos = None
        self.visited_positions = []

        self.visited_boxes = set()

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
        self.steps_since_reset = 0
        return np.array(init_state).astype(np.float32), {}


        #Calcula a reward
    def reward(self, action):
        reward = 0
        terminated = False
        truncated = False

        proximity_sensors = self.get_proximity_sensors()
        ground_sensors = self.get_ground_sensors()
        current_pos = self.get_robot_position()

        if self.enable_ground_reward:
            # Evitar quedas do cenário
            if min(ground_sensors) < 0.68:
                reward -=5
                terminated = True
                return reward, terminated,truncated

        # Recompensa negativa por colisões
        if max(sensor.getValue() for sensor in self.proximity_sensors) > 0.92:
            if self.enable_collision_reward:
                reward -= 5
                terminated = True
                return reward, terminated,truncated

        # Explorar o espaço
        if self.enable_movement_reward:
            if self.prev_pos is None:
                self.prev_pos = current_pos
                delta = 0
            else:
                delta = np.linalg.norm(current_pos - self.prev_pos)

            reward += 0.5 * delta



            if delta < 0.01:
                if not hasattr(self, "steps_stuck"):
                    self.steps_stuck = 1
                else:
                    self.steps_stuck += 1
            else:
                self.steps_stuck = 0

            if self.steps_stuck > 60:
                reward -= 3.0

            self.prev_pos = current_pos

        reward += self.check_box_visits()

        # Valorizar velocidades lineares positivas
        vel_left, vel_right = action
        vel_linear = (vel_left + vel_right) / 2.0

        if vel_linear > 0:
            reward += .6 * vel_linear  # avançar vale mais
        else:
            reward += 0.2 * abs(vel_linear)


        # Termina o episódio
        if self.steps_since_reset >= self.max_episode_steps:
            truncated = True

        return reward, terminated, truncated

    # Corre um Timestamp do teste
    def step(self, action, training=True):

        self.apply_action(action, training=training)

        for i in range(10):
            if Supervisor.step(self, self.__timestep) == -1:
                return self.state.astype(np.float32), 0, True, False, {}

        self.state = self.read_sensors()
        reward, terminated, truncated = self.reward(action)
        self.steps_since_reset += 1
        #print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Steps: {self.steps_since_reset}/{self.max_episode_steps}")

        return self.state.astype(np.float32), reward, terminated, truncated, {}


#PPO treino
def train_ppo():
    env = OpenAIGymEnvironment()
    env = Monitor(env)

    base_dir = "./thymio_models/ppo_def/"
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    best_model_dir = os.path.join(base_dir, "best_model")
    eval_logs_dir = os.path.join(base_dir, "eval_logs")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_logs_dir, exist_ok=True)

    # Inicializa o modelo
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-5,
        batch_size=128,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoints_dir,
        name_prefix='ppo_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_model_dir,
        log_path=eval_logs_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("Starting PPO training...")
    model.learn(
        total_timesteps=100000,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="run"
    )
    print("PPO training finished.")

    model.save(os.path.join(base_dir, "ppo_thymio_final"))

    with open(os.path.join(base_dir, "ppo_config.txt"), "w") as f:
        f.write(str(model.get_parameters()))

    env.close()





#Teste do modelo PPO
def test_ppo_model():
    print("Starting PPO model test...")
    env = OpenAIGymEnvironment()
    try:
        model = PPO.load("ppo_best_model/best_model")
    except Exception as e:
        print(f"ERROR: Could not load model 'ppo_thymio_final'. Make sure it exists. Error: {e}")
        return

    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

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

# Treino do Recurrent PPO
def train_recurrent_ppo():
    env = OpenAIGymEnvironment()
    env = Monitor(env)

    base_dir = "./thymio_models/recurrent_ppo_def/"
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    best_model_dir = os.path.join(base_dir, "best_model")
    eval_logs_dir = os.path.join(base_dir, "eval_logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        batch_size=128,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.03,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoints_dir,
        name_prefix='recurrent_ppo_def_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_model_dir,
        log_path=eval_logs_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    with open(os.path.join(base_dir, "recurrent_ppo_def_config.txt"), "w") as f:
        f.write(str(model.get_parameters()))

    print("Starting Recurrent PPO training...")
    model.learn(
        total_timesteps=100000,
        callback=[checkpoint_callback],
        tb_log_name="run"
    )
    print("Recurrent PPO training finished.")

    model.save(os.path.join(base_dir, "recurrent_ppo_def_thymio_final"))

    env.close()

#Treino de um modelo Recurrent PPO anteriormente desenvolvido
def train_recurrent_ppo_continued(
        model_path: str,
        initial_learning_rate: float,
        total_timesteps_to_add: int,
        new_tensorboard_log_name: str = "recurrent_ppo_def_continued",
        save_prefix: str = "recurrent_ppo_def_thymio_continued"
):

    env = OpenAIGymEnvironment()
    if not os.path.exists(model_path):
        print(f"Erro: Modelo pré-treinado não encontrado em '{model_path}'. Por favor, verifique o caminho.")
        env.close()
        return

    checkpoints_save_path = f'./{save_prefix}_checkpoints/'
    best_model_save_path = f'./{save_prefix}_best_model/'
    eval_logs_path = f'./{save_prefix}_eval_logs/'
    tensorboard_log_path = "./recurrent_ppo_def_continued_tensorboard/"
    tensorboard_root_log_path = "./tensorboard_logs_all_runs/"

    os.makedirs(checkpoints_save_path, exist_ok=True)
    os.makedirs(best_model_save_path, exist_ok=True)
    os.makedirs(eval_logs_path, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)

    def linear_learning_rate_schedule(progress_remaining: float) -> float:
        return initial_learning_rate * progress_remaining


    model = RecurrentPPO.load(
        path=model_path,
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_root_log_path

    )

    model.learning_rate = linear_learning_rate_schedule


    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoints_save_path,
        name_prefix=save_prefix
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_model_save_path,
        log_path=eval_logs_path,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    print("Starting Recurrent PPO training...")
    model.learn(
        total_timesteps=total_timesteps_to_add,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False,
        tb_log_name=new_tensorboard_log_name,
    )
    print("Recurrent PPO training finished.")


    final_model_name = f"{save_prefix}_final.zip"
    model.save(final_model_name)

    env.close()


#Teste do recurrent PPO
def test_recurrent_ppo_model():
    print("Starting Recurrent PPO model test...")
    env = OpenAIGymEnvironment()
    try:
        model = RecurrentPPO.load("recurrent_ppo_def_thymio_final.zip")
    except Exception as e:
        print(f"ERROR: Could not load model 'recurrent_ppo_thymio_final'. Make sure it exists. Error: {e}")
        return

    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    state = None
    episode_start = True

    while not done:
        action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
        print(f"Test Step: {steps}, Action: [{action[0]:.2f}, {action[1]:.2f}], Reward: {reward:.2f}, Total Episode Reward: {episode_reward:.2f}")
        episode_start = False

        if steps >= env.spec.max_episode_steps:
            print(f"Episode reached max steps ({env.spec.max_episode_steps}). Terminating.")
            done = True

    print(f"Test episode finished. Total Reward: {episode_reward:.2f}, Total Steps: {steps}")
    print("Recurrent PPO model test concluded.")

    env.close()



#TESTES com Relu e Tanh

#PPO treino com Relu e Tanh
def train_ppo_relu_tanh():
    """ env = OpenAIGymEnvironment()
    if not os.path.exists('./ppo_relu_checkpoints'):
        os.makedirs('./ppo_relu_checkpoints')


    policy_kwargs_relu = dict(activation_fn=nn.ReLU)
    model_relu = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs_relu,
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
        tensorboard_log="./ppo_relu_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./ppo_relu_checkpoints/',
        name_prefix='ppo_relu_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./ppo_relu_best_model/',
        log_path='./ppo_relu_eval_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting PPO Relu training...")
    model_relu.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback], tb_log_name="ppo_relu")
    print("PPO Relu training finished.")

    model_relu.save("ppo_relu_thymio_final")

    env.close()

        """
    env = OpenAIGymEnvironment()
    if not os.path.exists('./ppo_tanh_checkpoints'):
        os.makedirs('./ppo_tanh_checkpoints')

    policy_kwargs_tanh = dict(activation_fn=nn.Tanh)

    model_tanh = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs_tanh,
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
        tensorboard_log="./ppo_tanh_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./ppo_tanh_checkpoints/',
        name_prefix='ppo_tanh_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./ppo_tanh_best_model/',
        log_path='./ppo_tanh_eval_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting PPO Tanh training...")
    model_tanh.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback],tb_log_name="ppo_tanh")
    print("PPO Tanh training finished.")

    model_tanh.save("ppo_tanh_thymio_final")

    env.close()




#Treino do Recurrent PPO Tahn e Relu
def train_recurrent_ppo_tahn_relu():
    """env = OpenAIGymEnvironment()

    if not os.path.exists('./recurrent_ppo_relu_checkpoints'):
        os.makedirs('./recurrent_ppo_relu_checkpoints')

    policy_kwargs_relu = dict(activation_fn=nn.ReLU)



    model_relu = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs_relu,
        verbose=1,
        learning_rate=5e-5,
        batch_size=64,
        n_steps=256,
        gamma=0.99,
        ent_coef=0.05,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./recurrent_ppo_relu_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./recurrent_ppo_relu_checkpoints/',
        name_prefix='recurrent_ppo_relu_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./recurrent_ppo_relu_best_model/',
        log_path='./recurrent_ppo_relu_eval_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting Recurrent PPO Relu training...")
    model_relu.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback],tb_log_name="recurrent_ppo_relu")
    print("Recurrent PPO Relu training finished.")

    model_relu.save("recurrent_ppo_relu_thymio_final")

    env.close()
    """

    env = OpenAIGymEnvironment()

    if not os.path.exists('./recurrent_ppo_tanh_checkpoints'):
        os.makedirs('./recurrent_ppo_tanh_checkpoints')

    policy_kwargs_tanh = dict(activation_fn=nn.Tanh)



    model_tanh = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs_tanh,
        verbose=1,
        learning_rate=5e-5,
        batch_size=64,
        n_steps=256,
        gamma=0.99,
        ent_coef=0.05,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./recurrent_ppo_tanh_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./recurrent_ppo_tanh_checkpoints/',
        name_prefix='recurrent_ppo_tanh_thymio'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./recurrent_ppo_tanh_best_model/',
        log_path='./recurrent_ppo_tanh_eval_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting Recurrent PPO Tanh training...")
    model_tanh.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback], tb_log_name="recurrent_ppo_tanh")
    print("Recurrent PPO Tanh training finished.")

    model_tanh.save("recurrent_ppo_tanh_thymio_final")

    env.close()


#TESTES com elementos de recompensas


def _train_ppo_rewards(
        scenario_name,
        enable_ground_reward=True,
        enable_collision_reward=True,
        enable_movement_reward=True,
        enable_linear_vel_reward=True,
        randomize_on_reset=True,
):
    print(f"\n--- Starting PPO training: {scenario_name} ---")

    env = OpenAIGymEnvironment(
        enable_ground_reward=enable_ground_reward,
        enable_collision_reward=enable_collision_reward,
        enable_movement_reward=enable_movement_reward,
        enable_linear_vel_reward=enable_linear_vel_reward,
        randomize_on_reset = randomize_on_reset
    )

    log_dir_prefix = f"./ppo_ablation_logs/{scenario_name.replace(' ', '_').lower()}"
    model_save_dir = f"./ppo_ablation_models/{scenario_name.replace(' ', '_').lower()}"

    os.makedirs(log_dir_prefix, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(f"{model_save_dir}/eval_logs", exist_ok=True)
    os.makedirs(f"{model_save_dir}/best_model", exist_ok=True)


    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-5,
        batch_size=32,
        n_steps=1024,
        gamma=0.99,
        ent_coef=0.05,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir_prefix
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{model_save_dir}/checkpoints/",
        name_prefix=f'ppo_{scenario_name.replace(" ", "_").lower()}'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{model_save_dir}/best_model/",
        log_path=f"{model_save_dir}/eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    try:
        model.learn(total_timesteps=100000, callback=[checkpoint_callback, eval_callback], tb_log_name=scenario_name)
    except Exception as e:
        print(f"Error during {scenario_name} training: {e}")
    finally:
        print(f"{scenario_name} training finished.")
        model.save(f"{model_save_dir}/final_model")
        env.close()


def train_ppo_no_randomize():
    _train_ppo_rewards("No Randomize", randomize_on_reset=False)

def train_ppo_no_ground_penalty():
    _train_ppo_rewards("No Ground Penalty", enable_ground_reward=False)

def train_ppo_no_collision_penalty():
    _train_ppo_rewards("No Collision Penalty", enable_collision_reward=False)

def train_ppo_no_movement_reward():
    _train_ppo_rewards("No Movement Reward", enable_movement_reward=False)

def train_ppo_no_linear_vel_reward():
    _train_ppo_rewards("No Linear Velocity Reward", enable_linear_vel_reward=False)
def train_ppo_reward():
    _train_ppo_rewards("All rewards")


#Treino com diferentes reward retirados
def _train_recurrent_ppo_rewards(
        scenario_name,
        enable_ground_reward=True,
        enable_collision_reward=True,
        enable_movement_reward=True,
        enable_linear_vel_reward=True,
        randomize_on_reset=True,

):
    print(f"\n--- Starting Recurrent PPO training: {scenario_name} ---")

    env = OpenAIGymEnvironment(
        enable_ground_reward=enable_ground_reward,
        enable_collision_reward=enable_collision_reward,
        enable_movement_reward=enable_movement_reward,
        enable_linear_vel_reward=enable_linear_vel_reward,
        randomize_on_reset = randomize_on_reset
    )

    log_dir_prefix = f"./recurrent_ppo_ablation_logs/{scenario_name.replace(' ', '_').lower()}"
    model_save_dir = f"./recurrent_ppo_ablation_models/{scenario_name.replace(' ', '_').lower()}"

    os.makedirs(log_dir_prefix, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(f"{model_save_dir}/eval_logs", exist_ok=True)
    os.makedirs(f"{model_save_dir}/best_model", exist_ok=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=5e-5,
        batch_size=64,
        n_steps=256,
        gamma=0.99,
        ent_coef=0.05,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir_prefix,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{model_save_dir}/checkpoints/",
        name_prefix=f'ppo_{scenario_name.replace(" ", "_").lower()}'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{model_save_dir}/best_model/",
        log_path=f"{model_save_dir}/eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    try:
        model.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback], tb_log_name=scenario_name)
    except Exception as e:
        print(f"Error during {scenario_name} training: {e}")
    finally:
        print(f"{scenario_name} training finished.")
        model.save(f"{model_save_dir}/final_model")
        env.close()





def train_recurrent_ppo_no_randomize():
    _train_recurrent_ppo_rewards("No Randomize", randomize_on_reset=False)

def train_recurrent_ppo_no_ground_penalty():
    _train_recurrent_ppo_rewards("No Ground Penalty", enable_ground_reward=False)

def train_recurrent_ppo_no_collision_penalty():
    _train_recurrent_ppo_rewards("No Collision Penalty", enable_collision_reward=False)

def train_recurrent_ppo_no_movement_reward():
    _train_recurrent_ppo_rewards("No Movement Reward", enable_movement_reward=False)

def train_recurrent_ppo_no_linear_vel_reward():
    _train_recurrent_ppo_rewards("No Linear Velocity Reward", enable_linear_vel_reward=False)




#
#Para ver gráficos façam: ...\Proj1\Base Parte 2\thymio_rl_template\controllers\thymio_rl_controller> tensorboard --logdir .
# com o vosso caminho
#
def main():
    #train_recurrent_ppo_no_randomize()
    #train_recurrent_ppo_no_ground_penalty()
    #train_recurrent_ppo_no_collision_penalty()
    #train_recurrent_ppo_no_movement_reward()
    #train_recurrent_ppo_no_linear_vel_reward()


    #train_ppo_reward()
    #train_ppo_no_randomize()
    #train_ppo_no_ground_penalty()
    #train_ppo_no_collision_penalty()
    #train_ppo_no_movement_reward()
    #train_ppo_no_linear_vel_reward()


    #train_ppo_relu_tanh()
    #train_recurrent_ppo_tahn_relu()


    train_ppo()
    #test_ppo_model()
    #train_recurrent_ppo()
    #train_recurrent_ppo_continued("recurrent_ppo_def_thymio_final.zip",5e-5 , 1000000 )
    #test_recurrent_ppo_model()



if __name__ == '__main__':
    main()