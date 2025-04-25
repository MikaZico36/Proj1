import numpy as np
from controller import Supervisor
import random
import math
import numpy as np

# Simulation parameters
TIME_STEP = 5
POPULATION_SIZE = 100
PARENTS_KEEP = 3
INPUT = 5
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (1+INPUT)*HIDDEN  + (HIDDEN+1)*OUTPUT
GENERATIONS = 20
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300  # Simulated seconds per individual
RANGE = 5
WEIGHTS = 6

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
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)

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

        self.__n = 0
        self.prev_position = self.supervisor.getSelf().getPosition()
        
        
   

    def reset(self, seed=None, options=None):
        
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 0])


        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        

    def runStep(self, weights):
        self.collision = bool(
                self.__n > 10 and
                (self.__ir_0.getValue()>4300 or 
                self.__ir_1.getValue()>4300 or
                self.__ir_2.getValue()>4300 or
                self.__ir_3.getValue()>4300 or
                self.__ir_4.getValue()>4300 or
                self.__ir_5.getValue()>4300 or
                self.__ir_6.getValue()>4300)
            )
        
        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3

        print(f"GROUND_SENSOR_LEFT: {self.ground_sensors[0].getValue()}")

        left_speed =  ground_sensor_left * weights[0] + ground_sensor_right * weights[1] + weights[2]
        right_speed = ground_sensor_left * weights[3] + ground_sensor_right * weights[4] + weights[5]
        
        self.left_motor.setVelocity(max(min(left_speed, 9), -9))
        self.right_motor.setVelocity(max(min(right_speed, 9), -9))

        self.supervisor.step(self.timestep)


    def runBest(self, weights):
        self.reset()
        self.evaluation_start_time = self.supervisor.getTime()
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            self.runStep(weights)


   
    def run(self):
        self.evaluation_start_time = self.supervisor.getTime()
        weights = [0.9962394580264582, -0.31073551918942766, 0.16609185696341, -0.6785534060759024, 0.9943027615824647, 0.6560064809304231]
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            self.runStep(weights)

#Corre um robô durante o tempo estipulado e a cada movimento calcula o seu fitness. No final, retorna o fitness calculado
    def runRobot(self, weights):

        fitness = 0
        self.reset()
        self.evaluation_start_time = self.supervisor.getTime()
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:



            ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3
            ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3


            fitness = calculate_fitness(ground_sensor_left, ground_sensor_right,fitness,self)

            #left_speed =  ground_sensor_left * weights[0] + ground_sensor_right * weights[1] + weights[2]
            #right_speed = ground_sensor_left * weights[3] + ground_sensor_right * weights[4] + weights[5]

            left_speed = (weights[0]*ground_sensor_left)+ (weights[1]*ground_sensor_right)+ weights[2]
            right_speed = (weights[3]*ground_sensor_left)+ (weights[4]*ground_sensor_right)+ weights[5]



            self.left_motor.setVelocity(max(min(left_speed, 9), -9))
            self.right_motor.setVelocity(max(min(right_speed, 9), -9))

            self.supervisor.step(self.timestep)
        return fitness

        
       
            

#Cria um vetor com os pesos da população
def initialize_population():
    return [{'weights': np.random.uniform(-1, 1, WEIGHTS), 'fitness': 0} for _ in range(POPULATION_SIZE)]


def sorted_parents(population):
    return sorted(population, key=lambda x: x['fitness'], reverse=True)[:25]

#Faz o torneio entre dois indivíduo e retorna aquele com mais fitness
def tournament(p1,p2):
    return p1 if p1['fitness'] >= p2['fitness'] else p2



#Faz o crossover num ponto aleatório dos indivíduos e ainda aplica mutation aos filhos
def crossover(population):

    new_population = []
    best_population = sorted_parents(population)

    for i in range(best_population.__len__()):
        p1,p2,p3,p4 = random.sample(best_population, 4)

        winner1 = tournament(p1,p2)
        winner2 = tournament(p3,p4)

        for _ in range(2):
            crossover_point = random.randint(1, WEIGHTS - 1)

            child1_weights = np.concatenate((winner1['weights'][:crossover_point], winner2['weights'][crossover_point:]))
            child2_weights = np.concatenate((winner2['weights'][:crossover_point], winner1['weights'][crossover_point:]))

            child1 = {'weights': child1_weights, 'fitness': 0}
            child2 = {'weights': child2_weights, 'fitness': 0}

            mutated_child1 = mutate(child1)
            mutated_child2 = mutate(child2)

            new_population.append(mutated_child1)
            new_population.append(mutated_child2)




    return new_population


#Faz a mutaçao de um ou mais genes no indivíuo
def mutate(individual):
    for i in range(WEIGHTS):
        if random.random() < MUTATION_RATE:
            individual['weights'][i] += np.random.normal(0, MUTATION_SIZE)
    return individual

#calcula o valor do fitness- Quanto mais tempo o robô estiver sobre a linha preta mais fitness recebe
#Agora também calcula a distância a que está do centro. Para valorizar uma exploração maior.
def calculate_fitness(left_sensor,right_sensor, fitness,self):

    #position = self.robot_node.getPosition()
    #distance = np.linalg.norm(position)

    #fitness += distance/1000


    if not left_sensor and not right_sensor:
        fitness += 10
    elif not left_sensor or not right_sensor:
        fitness += 1
    return fitness

def main2():
    controller = Evolution()
    controller.run()


# Main evolutionary loop
def main():

    # Run the evolutionary algorithm
    controller = Evolution()
    population = initialize_population()

    best_population = []


    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation+1}")

        for individual in population:
            individual['fitness'] = controller.runRobot(individual['weights'])
            print(f"\n Fitness: {individual['fitness']}")

        sorted = sorted_parents(population)
        print(sorted)
        print(f"Best fitness: {sorted[0]['fitness']}")



        new_population = crossover(population)

        population = new_population

        """with open("melhores_individuos.txt", "a") as f:
            f.write(f"--- Geração {generation+1} ---\n")
            for i, ind in enumerate(best_population):
                f.write(f"Indivíduo {i+1}:\n")
                f.write(f"Fitness: {ind['fitness']}\n")
                f.write(f"Weights: {ind['weights'].tolist()}\n\n")
            f.write("\n\n")
"""
    print(sorted_parents(population)[0])


if __name__ == "__main__":
    main()

