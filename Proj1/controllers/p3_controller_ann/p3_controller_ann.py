import numpy as np
from controller import Supervisor
import random
import numpy as np

# Simulation parameters
TIME_STEP = 5
POPULATION_SIZE = 100
PARENTS_KEEP = 3
INPUT = 5
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (1+INPUT)*HIDDEN  + (HIDDEN+1)*OUTPUT
GENERATIONS = 10
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 100  # Simulated seconds per individual
RANGE = 5
WEIGHTS = 6
ANN_PARAMS = 54


def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return (0, 0, 1, angle)

def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = random_orientation()
    x = radius * np.cos(angle[3])
    y = radius * np.sin(angle[3])
    return (x, y, z)

class Evolution:
    def __init__(self):
        self.evaluation_start_time = 0
        self.collision = False

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()

        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep()*TIME_STEP)
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.__ir_0 = self.supervisor.getDevice('prox.horizontal.0')
        self.__ir_1 = self.supervisor.getDevice('prox.horizontal.1')
        self.__ir_2 = self.supervisor.getDevice('prox.horizontal.2')
        self.__ir_3 = self.supervisor.getDevice('prox.horizontal.3')
        self.__ir_4 = self.supervisor.getDevice('prox.horizontal.4')
        self.__ir_5 = self.supervisor.getDevice('prox.horizontal.5')
        self.__ir_6 = self.supervisor.getDevice('prox.horizontal.6')
        self.__ir_7 = self.supervisor.getDevice('prox.ground.0')
        self.__ir_8 = self.supervisor.getDevice('prox.ground.1')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.__ir_0.enable(self.timestep)
        self.__ir_1.enable(self.timestep)
        self.__ir_2.enable(self.timestep)
        self.__ir_3.enable(self.timestep)
        self.__ir_4.enable(self.timestep)
        self.__ir_5.enable(self.timestep)
        self.__ir_6.enable(self.timestep)
        self.__ir_7.enable(self.timestep)
        self.__ir_8.enable(self.timestep)

        self.sensors = [self.__ir_0,self.__ir_2,self.__ir_4]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        self.horizontal_sensors = [self.supervisor.getDevice(f'prox.horizontal.{i}') for i in [0,2,4]]

        self.__n = 0
        self.prev_position = self.supervisor.getSelf().getPosition()



    def run(self, weights):
        fitness = 0
        self.reset()
        self.evaluation_start_time = self.supervisor.getTime()
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2
            ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2
            horizontal_sensor_central = (self.horizontal_sensors[0].getValue()/1023 - .6)/.2
            horizontal_sensor_right = (self.horizontal_sensors[1].getValue()/1023 - .6)/.2
            horizontal_sensor_left = (self.horizontal_sensors[2].getValue()/1023 - .6)/.2


            inputs = [ground_sensor_left,
                      ground_sensor_right,
                      horizontal_sensor_central,
                      horizontal_sensor_right,
                      horizontal_sensor_left]

            outputs = ann_forward(weights, [ground_sensor_left, ground_sensor_right, horizontal_sensor_central, horizontal_sensor_right, horizontal_sensor_left])
            fitness = calculate_fitness(inputs, fitness, self)

            left_speed = outputs[0]*9
            right_speed = outputs[1]*9

            self.left_motor.setVelocity(max(min(left_speed, 9), -9))
            self.right_motor.setVelocity(max(min(right_speed, 9), -9))

            self.supervisor.step(self.timestep)
        return fitness

    def is_inside_obstacle(pos, obstacle_pos, obstacle_size, margin=0.05):
        return (abs(pos[0] - obstacle_pos[0]) < (obstacle_size[0] / 2 + margin)) and \
            (abs(pos[1] - obstacle_pos[1]) < (obstacle_size[1] / 2 + margin))



    def reset(self, seed=None, options=None):
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 0])

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


def inicialize_population_ann():
    return [{'weights': np.random.uniform(-1, 1, ANN_PARAMS), 'fitness': 0} for _ in range(POPULATION_SIZE)]


def ann_forward(weights, inputs):
    i = 0
    w1 = np.array(weights[i: i + INPUT * HIDDEN]).reshape((INPUT, HIDDEN))
    i += INPUT * HIDDEN
    b1 = np.array(weights[i: i + HIDDEN])
    i += HIDDEN

    w2 = np.array(weights[i: i + HIDDEN * HIDDEN]).reshape((HIDDEN, HIDDEN))
    i += HIDDEN * HIDDEN
    b2 = np.array(weights[i: i + HIDDEN])
    i += HIDDEN

    w3 = np.array(weights[i: i + HIDDEN * OUTPUT]).reshape((HIDDEN, OUTPUT))
    i += HIDDEN * OUTPUT
    b3 = np.array(weights[i:i + OUTPUT])

    hidden1 = np.tanh(np.dot(inputs, w1) + b1)
    hidden2 = np.tanh(np.dot(hidden1, w2) + b2)
    output = np.tanh(np.dot(hidden2, w3) + b3)

    return output


def crossover_ann(population):
    new_population = []
    best_fitness = sorted(population, key=lambda x: x['fitness'], reverse=True)[:PARENTS_KEEP]
    new_population.extend(best_fitness)
    while len(new_population) < POPULATION_SIZE:
        p1, p2 = random.sample(population, 2)
        point = random.randint(1, ANN_PARAMS - 1)
        c1_weights = np.concatenate((p1['weights'][:point], p2['weights'][point:]))
        c2_weights = np.concatenate((p2['weights'][:point], p1['weights'][point:]))
        c1 = mutate_ann({'weights': c1_weights, 'fitness': 0})
        c2 = mutate_ann({'weights': c2_weights, 'fitness': 0})
        new_population.extend([c1, c2])

    return new_population[:POPULATION_SIZE]


def mutate_ann(individual):
    for i in range(ANN_PARAMS):
        if random.random() < MUTATION_RATE:
            individual['weights'][i] += np.random.normal(0, MUTATION_SIZE)
    return individual

def calculate_fitness(inputs, fitness, self):
    if inputs[2] > 1.2 or inputs[3] > 1.2 or inputs[4] > 1.2:
        fitness += -5
    if inputs[0] < -2 and inputs[1] < -2:
        fitness += 10
    elif inputs[0] < -2 or inputs[1] < -2:
        fitness += 5


    return fitness


def sorted_parents(population):
    return sorted(population, key=lambda x: x['fitness'], reverse=True)[:25]

def main():
    controller = Evolution()
    population = inicialize_population_ann()

    best_population = []

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation+1}")
        for individual in population:
            individual['fitness'] = controller.run(individual['weights'])
            print(f"\n Fitness: {individual['fitness']}")

        population_sorted = sorted_parents(population)
        print(f"Best fitness: {population_sorted[0]['fitness']}")
        best_population.append(population_sorted[0])

        with open("melhores_individuos.txt", "a") as f:
            f.write(f"--- Geração {generation+1} ---\n")
            for i, ind in enumerate(best_population):
                f.write(f"Indivíduo {i+1}:\n")
                f.write(f"Fitness: {ind['fitness']}\n")
                f.write(f"Weights: {ind['weights'].tolist()}\n\n")
                f.write("\n\n")

        new_population = crossover_ann(population)
        population = new_population



if __name__ == "__main__":
    main()