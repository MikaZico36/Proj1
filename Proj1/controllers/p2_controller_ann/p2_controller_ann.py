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
GENERATIONS = 10
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300  # Simulated seconds per individual
RANGE = 5
WEIGHTS = 6
ANN_PARAMS = 22

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
        any(sensor.getValue() > 4300 for sensor in self.sensors)
    )

        inputs = [sensor.getValue()/1023 for sensor in self.ground_sensors]
        motor_speeds = ann_forward(weights, inputs)

        self.left_motor.setVelocity(max(min(motor_speeds[0]*9, 9), -9))
        self.right_motor.setVelocity(max(min(motor_speeds[1]*9, 9), -9))

        self.supervisor.step(self.timestep)



   
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


            fitness = calculate_fitness(ground_sensor_left, ground_sensor_right,fitness)

            #left_speed =  ground_sensor_left * weights[0] + ground_sensor_right * weights[1] + weights[2]
            #right_speed = ground_sensor_left * weights[3] + ground_sensor_right * weights[4] + weights[5]

            left_speed = (weights[0]*ground_sensor_left)+ (weights[1]*ground_sensor_right)+ weights[2]
            right_speed = (weights[3]*ground_sensor_left)+ (weights[4]*ground_sensor_right)+ weights[5]



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
    print(f"Output: {output}")
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
       
#calcula o valor do fitness- Quanto mais tempo o robô estiver sobre a linha preta mais fitness recebe
def calculate_fitness(left_sensor,right_sensor, fitness):
    if not left_sensor and not right_sensor:
        fitness += 2
    elif not left_sensor or not right_sensor:
        fitness +=1
    return fitness

def main2():
    controller = Evolution()
    controller.run()


# Main evolutionary loop
def main():
    controller = Evolution()
    population = inicialize_population_ann()
    
    best_population = []

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation+1}")

        for individual in population:
            individual['fitness'] = controller.runRobot(individual['weights'])
            print(f"\n Fitness: {individual['fitness']}")
            
            if individual['fitness']> 2000:
                best_population.append(individual)
        population_sorted = sorted_parents(population)
        print(f"Best fitness: {population_sorted[0]['fitness']}")
        new_population = crossover_ann(population)
        population = new_population
        


if __name__ == "__main__":
    main()

