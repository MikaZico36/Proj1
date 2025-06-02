#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

try:
    import time
    import gymnasium as gym
    import numpy as np
    import math
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps = ...):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', entry_point='openai_gym:OpenAIGymEnvironment', max_episode_steps=max_episode_steps)
        self.__timestep = int(self.getBasicTimeStep())

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([...]),
            high=np.array([...]), dtype=np.float32)

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
        self.observation_space = gym.spaces.Box(
            low=np.array([...]),
            high=np.array([...]), dtype=np.float32)

        self.state = None

        # Do all other required initializations
        ...


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
        ...

        # you may need to iterate a few times to let physics stabilize
        for i in range(15):
            super().step(self.__timestep)

        # set the initial state vector to return
        init_state = ...

        return np.array(init_state).astype(np.float32), {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #
    def step(self, action):

        self.__n = self.__n + 1

        # start by applying the action in the robot actuators
        # See how in Lab 1 code
        ...

        # let the action to effect for a few timesteps
        for i in range(10):
            super().step(self.__timestep)


        # set the state that resulted from applying the action (consulting the robot sensors)
        self.state = np.array([ ... ])

        # compute the reward that results from applying the action in the current state
        reward = ...

        # set termination and truncation flags (bools)
        terminated = ...
        truncated = ...

        return self.state.astype(np.float32), reward, terminated, truncated, {}


def main():

    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()

    # Code to train and save a model
    # For the PPO case, see how in Lab 7 code
    # For the RecurrentPPO case, consult its documentation
    ...

    # Code to load a model and run it
    # For the RecurrentPPO case, consult its documentation
    ...


if __name__ == '__main__':
    main()
