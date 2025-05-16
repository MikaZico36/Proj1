import numpy as np
from controller import Supervisor
import random
import math
import numpy as np

# Simulation parameters
TIME_STEP = 5
POPULATION_SIZE = 20
PARENTS_KEEP = 3
INPUT = 5
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (1+INPUT)*HIDDEN  + (HIDDEN+1)*OUTPUT
GENERATIONS = 1000
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300 # Simulated seconds per individual
RANGE = 5
WEIGHTS = 6
ANN_PARAMS = 22

def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return [0, 0, 1, angle] 

def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = np.random.uniform(0, 2 * np.pi)  # Corrigido para retornar apenas o ângulo
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return [x, y, z]  # Deve retornar uma lista com 3 elementos

class Evolution:
    def __init__(self):
        self.evaluation_start_time = 0
        self.collision = False
        self.visited_areas = set()  # Rastrear áreas visitadas
        self.last_time_update = 0  # Rastrear o último tempo de atualização do fitness

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()

        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep() * TIME_STEP)
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

        self.sensors = [self.__ir_0, self.__ir_2, self.__ir_4]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        self.__n = 0
        self.prev_position = self.supervisor.getSelf().getPosition()
        self.time_on_line = 0

    def reset(self, seed=None, options=None):
        random_rotation = random_orientation() 
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)

        random_pos = random_position(0.2, 0.5, 0)  
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f(random_pos)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def run(self):
        self.evaluation_start_time = self.supervisor.getTime()
        #best_weights =  [0.5415714249713564, 0.26886237548295844, 0.7761207874495106, -0.4260348739874109, -0.9908881795897919, 0.6643106583749459, -0.4826656248145633, 0.6033248365002477, 0.8826901566888914, 0.8458925161992736, 0.3197980221884944, 0.24555656921134927, 0.9424106648448496, -0.5551927634647345, -0.2091814311937461, 0.7207433516262489, -0.3621457799519938, 0.8783554801291444, 0.7453287109605891, 0.21812111044613047, 0.7015652238157566, 0.6553864364634756]
        weights = get_weights() 
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            self.runRobot(weights)

#Corre um robô durante o tempo estipulado e a cada movimento calcula o seu fitness. No final, retorna o fitness calculado
    def runRobot(self, weights):
        fitness = 0
        self.reset()
        self.evaluation_start_time = self.supervisor.getTime()
        self.time_on_line = 0  # Resetar contador no início da execução
        self.visited_areas.clear()  # Limpar áreas visitadas
        self.last_time_update = self.evaluation_start_time  # Resetar tempo de atualização

        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            ground_sensor_left = (self.ground_sensors[0].getValue() / 1023 - .6) / .2
            ground_sensor_right = (self.ground_sensors[1].getValue() / 1023 - .6) / .2

            outputs = ann_forward(weights, [ground_sensor_left, ground_sensor_right])
            fitness, self.last_time_update = calculate_fitness(
                ground_sensor_left, ground_sensor_right, fitness, self.supervisor, self.visited_areas, self.last_time_update
            )

            left_speed = outputs[0] * 9
            right_speed = outputs[1] * 9

            self.left_motor.setVelocity(max(min(left_speed, 9), -9))
            self.right_motor.setVelocity(max(min(right_speed, 9), -9))

            self.supervisor.step(self.timestep)

        return fitness

def inicialize_population_ann():
    return [{'weights': np.random.uniform(-1, 1, ANN_PARAMS), 'fitness': 0} for _ in range(POPULATION_SIZE)]

def ann_forward(weights, inputs):
    w1 = np.array(weights[0:8]).reshape((2, 4))      
    b1 = np.array(weights[8:12])                   
    w2 = np.array(weights[12:20]).reshape((4, 2))    
    b2 = np.array(weights[20:22])                  

    hidden = np.tanh(np.dot(inputs, w1) + b1)
    output = np.tanh(np.dot(hidden, w2) + b2)
    return output

def sorted_parents(population):
    return sorted(population, key=lambda x: x['fitness'], reverse=True)[:PARENTS_KEEP]

def crossover_ann(population):
    new_population = []
    for _ in range(POPULATION_SIZE//2):
        p1, p2 = random.sample(population, 2)
        point = random.randint(1, ANN_PARAMS - 1)
        c1_weights = np.concatenate((p1['weights'][:point], p2['weights'][point:]))
        c2_weights = np.concatenate((p2['weights'][:point], p1['weights'][point:]))
        c1 = mutate_ann({'weights': c1_weights, 'fitness': 0})
        c2 = mutate_ann({'weights': c2_weights, 'fitness': 0})
        new_population.extend([c1, c2])
    return new_population

def mutate_ann(individual):
    for i in range(ANN_PARAMS):
        if random.random() < MUTATION_RATE:
            individual['weights'][i] += np.random.normal(0, MUTATION_SIZE)
    return individual  
       
def calculate_fitness(left_sensor, right_sensor, fitness, supervisor, visited_areas, last_time_update):
    current_time = supervisor.getTime()

    if left_sensor < 0 and right_sensor < 0:
        if current_time - last_time_update >= 1:  
            fitness += 5
            last_time_update = current_time

    robot_position = supervisor.getSelf().getPosition()
    x, y, _ = robot_position

    black_areas = {
        "BlackArea": (1.0, 1.0, 0.25, 0.25),
        "BlackArea(1)": (0.99, 0.03, 0.1, 1.8),
        "BlackArea(2)": (0.05, 0.99, 1.8, 0.1),
        "BlackArea(3)": (-0.04, -1.08, 1.6, 0.1),
        "BlackArea(4)": (-1.06, -0.03, 0.1, 1.6),
        "BlackArea(5)": (-0.952341, 0.882337, 0.1, 0.38),
        "BlackArea(6)": (0.857858, -0.970266, 0.1, 0.38),
        "BlackArea(7)": (-0.945254, -0.949082, 0.1, 0.38)
    }

    for area_name, (ax, ay, w, h) in black_areas.items():
        if (ax - w/2 <= x <= ax + w/2) and (ay - h/2 <= y <= ay + h/2):
            if area_name not in visited_areas:
                visited_areas.add(area_name)
                fitness += 200
                break  

    if len(visited_areas) == 8:
        visited_areas.clear()

    return fitness, last_time_update

def get_weights():
    best_fitness = float('-inf')
    best_wights = []
    best_generation = 1
    with open("melhores_individuos.txt", 'r') as f:
        linhas = f.readlines()

    for i in range(len(linhas)):
        if linhas[i].startswith('--- Melhor Indivíduo da Geração'):
            partes = linhas[i].split()
            generation = int(partes[-2])

        if linhas[i].startswith("Fitness:"):
            fitness = int(linhas[i].split(":")[1].strip())

            if fitness > best_fitness:
                best_fitness = fitness
                best_generation = generation

                if i + 1 < len(linhas) and "Weights:" in linhas[i + 1]:
                    pesos_str = linhas[i + 1].split(":", 1)[1].strip()
                    best_wights = [float(p) for p in pesos_str.strip('[]').split(",")]
    
    print(best_generation)
    return best_wights

def main2():
    controller = Evolution()
    controller.run()


def main():
    controller = Evolution()
    population = inicialize_population_ann()

    with open("melhores_individuos.txt", "a") as f:
        for generation in range(GENERATIONS):
            print(f"\nGeneration {generation+1}")

            for individual in population:
                individual['fitness'] = controller.runRobot(individual['weights'])
                print(f"Fitness: {individual['fitness']}")

            population_sorted = sorted_parents(population)
            best_individual = population_sorted[0]
            print(f"Best fitness: {best_individual['fitness']}")

            f.write(f"--- Melhor Indivíduo da Geração {generation+1} ---\n")
            f.write(f"Fitness: {best_individual['fitness']}\n")
            f.write(f"Weights: {best_individual['weights'].tolist()}\n\n")

            # Gerar a próxima geração
            population = crossover_ann(population)


if __name__ == "__main__":
    main2()