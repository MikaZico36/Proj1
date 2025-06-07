import numpy as np
from controller import Supervisor
import random
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# Simulation parameters
TIME_STEP = 5
POPULATION_SIZE = 20
PARENTS_KEEP = 3
INPUT = 5
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (1+INPUT)*HIDDEN  + (HIDDEN+1)*OUTPUT
GENERATIONS = 1000
MUTATION_RATE = 0.3
MUTATION_SIZE = 0.4
EVALUATION_TIME = 200  # Simulated seconds per individual
RANGE = 5
WEIGHTS = 6
ANN_PARAMS = 140
NO_IMPROVEMENT_LIMIT = 40


def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return [0, 0, 1, angle]

def random_position(min_radius, max_radius, z):
    # Lista de posições dos cubos no formato [(x1, y1), (x2, y2), ...]
    cube_positions = [
        (0,0.3),
        (0.4,-0.7),
        (-0.5, -0.8), 
        (0.7, 0.7),
        (0.8,-0.5),
        (-0.7,0.8),
        (0.8,0),
        (-0.8,1.1)
    ]
    safe_distance = 0.5 

    while True:
        # Gera uma posição aleatória
        radius = np.random.uniform(min_radius, max_radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        valid_position = True
        for cx, cy in cube_positions:
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance < safe_distance:
                valid_position = False
                break

        if valid_position:
            return [x, y, z]

class Evolution:
    def __init__(self):
        self.evaluation_start_time = 0
        self.collision = False
        self.visited_areas = set()
        self.time_on_line = 0
        self.last_time_update = 0

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

    def run(self):
        self.evaluation_start_time = self.supervisor.getTime()
        weights = get_weights()
        best_weights = [0.39662396566289804, -0.13684176189310926, 0.4934398414108367, 1.2341611591226822, -0.38938517344125634, 1.075904231104372, 0.2153377218764001, 0.008717034130410806, -0.5307779202848688, -0.7954087997698653, 0.04312948554343948, 0.39018691874997624, 0.48408085739022466, 0.11939423215369616, -0.8224637392545635, 0.6196129482643331, 0.31214771665830976, 0.47883279886188795, 0.5711631180799241, -0.2790586325397546, -0.8501605344266983, -0.609253159058783, 1.1424586966035308, -0.8200413946589593, -0.48978959446814435, -0.6261217812644035, 1.4256442573356538, -0.18831601272827975, 0.42507802970933195, 0.36162171303219415, 0.8714128964319194, -0.4768367389493715, -0.2184174975060736, -0.70744836247934, 0.39086428824633884, -0.40918593915437107, 0.001770834928900733, -0.8772003570397966, -0.19821029139445145, 0.3286060049385069, -0.09208792567404539, 0.17209696654435258, -0.9382398195377156, 0.6736426899682935, 0.6389627573190361, -1.4699811296111147, -0.19930026040954874, -0.3915777834242198, -0.9189440236465614, 0.6086467353199536, -1.9746526967924387, 0.09449771500221815, 1.3991405504721144, 1.3668140583348567, 0.35714760499177256, 1.1061434879040117, 0.21102682966217423, 0.6538106710806848, -0.47243288165773656, 0.6122381982085758, -0.05420698989432893, 0.7099491523424911, 0.8655925322298474, 1.222331407592415, -0.19800545429416985, 1.1993733030527736, -0.6784524767396891, -0.06111351681554589, 0.569328295497852, 0.24673750074693035, -0.670996994058528, -0.6193931722625232, -0.2744499018953356, -0.9235998834420456, 0.7401476474009359, 0.9637078196011409, 0.6839254325630417, -0.2585793368586106, 1.6421145718155128, 1.2077819465090283, -0.9015849673445151, -0.09154662013952403, 0.24552162010479073, -0.9314522211706953, -0.10986299201655901, 0.11823838278113319, -1.0456679208623634, 0.34034485686535043, -0.0799778273545445, 0.3958657560750516, 0.4566580094998126, 1.054530503974125, 0.3748026298956838, 0.2577690702765578, 0.9665267438048697, -0.30566544602669754, -0.5511872236213461, 0.3950077028723004, -0.016030893968273313, -0.005729032721676908, -0.7545834166952401, 0.3922798267178791, -0.33343903015791654, -0.7305175342328041, -0.520866668301377, 0.7711710944409718, -0.9067523505234809, 0.41134944849472665, 0.8262771025115043, 0.9354651021856195, 1.2425651580246473, -0.2772092305056686, 0.8745961381815023, 0.5108764045011887, 0.48824595421251726, -0.6522036680894558, 0.4291254761682355, 0.6428933136468609, 1.2047304343242682, 0.38198022795299824, -0.6637877600191242, -0.016256403361762428, -1.1215239107598725, 0.9514526243331891, 0.5866408854505814, 0.5833940842102499, -0.2972803011581763, 0.7774163521971158, 0.4016292332731266, 0.23289506531170806, -0.9486126448220031, 1.340748088039537, -0.0937545344151125, 0.40238434056549416, -0.5018800338125298, -0.20405532277485672, 0.880996421274933, 1.420335116846109, 0.8755508873268749, -0.12561314745498553]
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            self.runRobot(best_weights)


    def runRobot(self, weights):
        fitness = 0
        self.reset()
        self.evaluation_start_time = self.supervisor.getTime()
        self.time_on_line = 0
        self.visited_areas.clear()
        self.last_time_update = self.evaluation_start_time  # Resetar tempo de atualização

        x, y, _ = self.supervisor.getSelf().getPosition()
        last_position = (x, y)

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

            fitness, self.last_time_update, last_position, stop = calculate_fitness(
                inputs, fitness, self, self.visited_areas, self.last_time_update, last_position
            )

            if stop:
                break
            
            outputs = ann_forward(weights, inputs)

            left_speed = outputs[0]*8
            right_speed = outputs[1]*8

            self.left_motor.setVelocity(max(min(left_speed, 9), -9))
            self.right_motor.setVelocity(max(min(right_speed, 9), -9))

            self.supervisor.step(self.timestep)

        return fitness



    def reset(self, seed=None, options=None):
        random_rotation = random_orientation() 
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)

        random_pos = random_position(0.8, 0.8, 0)
        center = [0,0,0]
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f(random_pos)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


def inicialize_population_ann():
    return [{'weights': np.random.uniform(-1, 1, ANN_PARAMS), 'fitness': 0} for _ in range(POPULATION_SIZE)]


def ann_forward(weights, inputs):
    i = 0

    # Camada 1: input -> hidden1 (8)
    w1 = np.array(weights[i: i + INPUT * 8]).reshape((INPUT, 8))
    i += INPUT * 8
    b1 = np.array(weights[i: i + 8])
    i += 8

    # Camada 2: hidden1 -> hidden2 (6)
    w2 = np.array(weights[i: i + 8 * 6]).reshape((8, 6))
    i += 8 * 6
    b2 = np.array(weights[i: i + 6])
    i += 6

    # Camada 3: hidden2 -> hidden3 (4)
    w3 = np.array(weights[i: i + 6 * 4]).reshape((6, 4))
    i += 6 * 4
    b3 = np.array(weights[i: i + 4])
    i += 4

    # Camada 4: hidden3 -> output (2)
    w4 = np.array(weights[i: i + 4 * 2]).reshape((4, 2))
    i += 4 * 2
    b4 = np.array(weights[i: i + 2])

    # Forward pass
    h1 = np.tanh(np.dot(inputs, w1) + b1)
    h2 = np.tanh(np.dot(h1, w2) + b2)
    h3 = np.tanh(np.dot(h2, w3) + b3)
    output = np.tanh(np.dot(h3, w4) + b4)

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

#ELEITISMO
def new_pop(population):
    new_population = []

    # Selecionar os 4 melhores indivíduos da geração anterior
    best_individuals = population[:4]
    new_population.extend(best_individuals)

    # Criar mutações dos melhores indivíduos para preencher o restante da população
    while len(new_population) < POPULATION_SIZE:
        for parent in best_individuals:
            if len(new_population) < POPULATION_SIZE:
                mutated_individual = mutate_ann({'weights': np.copy(parent['weights']), 'fitness': 0})
                new_population.append(mutated_individual)

    return new_population



def calculate_fitness(inputs, fitness, self, visited_areas, last_time_update, last_position):
    supervisor = self.supervisor
    current_time = supervisor.getTime()

    # Sensores de chão normalizados
    left_sensor = inputs[0]
    right_sensor = inputs[1]

    # Recompensa proporcional com base nos sensores de chão
    if left_sensor < 0 and right_sensor < 0:
        if current_time - last_time_update >= 1:
            fitness += 4
            last_time_update = current_time
    elif left_sensor < 0 or right_sensor < 0:
        if current_time - last_time_update >= 1:
            fitness += 1
            last_time_update = current_time

    # Sensores de proximidade frontais
    central = inputs[2]
    right = inputs[3]
    left = inputs[4]

    # Posição do robô (apenas x e y)
    robot_position = supervisor.getSelf().getPosition()
    x2, y2, _ = robot_position
    x1, y1 = last_position

    displacement = math.dist((x1, y1), (x2, y2))
    fitness += displacement * 2  # Recompensa movimento

    # Verificação de áreas pretas visitadas
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
        if (ax - w/2 <= x2 <= ax + w/2) and (ay - h/2 <= y2 <= ay + h/2):
            if area_name not in visited_areas:
                visited_areas.add(area_name)
                fitness += 200
                break

    # Verificar se todas as áreas pretas foram visitadas
    if len(visited_areas) == len(black_areas):
        self.time_on_line += 1  # Incrementar o contador de vezes que todas as áreas foram visitadas
        visited_areas.clear()  # Limpar as áreas visitadas para começar novamente

    # Parar o robô e atribuir fitness final se todas as áreas forem visitadas 3 vezes
    if self.time_on_line >= 2:
        fitness = 10000
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        print("Todas as áreas pretas foram visitadas 3 vezes. Fitness final: 10000")
        return fitness, last_time_update, (x2, y2), True  # Adiciona um sinalizador para parar o robô

    return fitness, last_time_update, (x2, y2), False  # Continua normalmente


def sorted_parents(population):
    return sorted(population, key=lambda x: x['fitness'], reverse=True)[:25]

def get_weights():
    best_fitness = float('-inf')
    best_wights = []
    best_generation = None
    generation = 0
    with open("melhores_individuos.txt", 'r') as f:
        linhas = f.readlines()

    for i in range(len(linhas)):
        
        if linhas[i].startswith("Fitness:"):
            fitness = float(linhas[i].split(":")[1].strip())

            if fitness > best_fitness:
                best_fitness = fitness
                best_generation = generation
                if i + 1 < len(linhas) and "Weights:" in linhas[i + 1]:
                    pesos_str = linhas[i + 1].split(":", 1)[1].strip()
                    best_wights = [float(p) for p in pesos_str.strip('[]').split(",")]
            generation += 1  
    print("Melhor fitness encontrado:", best_fitness)
    print("Melhor geração:", best_generation)
    return best_wights

def get_weights_to_pop():
    with open("melhores_individuos.txt", 'r') as f:
        linhas = f.readlines()
        individuos = []
        for i in range(len(linhas)):

            if linhas[i].startswith("Fitness:"):
                fitness = float(linhas[i].split(":")[1].strip())

            if linhas[i].startswith("Weights:"):
                weights_str = linhas[i].split(":")[1].strip()
                weights = [float(p) for p in weights_str.strip('[]').split(',')]
                individuos.append((fitness, weights))  # Apenas adiciona quando ambos estão disponíveis

        individuos = sorted(individuos, key=lambda x: x[0], reverse=True)[:POPULATION_SIZE]

        return [ind[1] for ind in individuos]


def have_20_individuals():
    with open("melhores_individuos.txt", 'r') as f:
        lines = f.readlines()
        fitness_count = 0
        for line in lines:
            if line.startswith("Fitness:"):
                fitness_count += 1
                if fitness_count >= 20:
                    return True
    return False


def plot_graph():
    generations = []
    generation = 0
    fitness = []

    with open("melhores_individuos.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Fitness:"):
                try:
                    fitness_value = float(line.split(":")[1].strip())
                    generations.append(generation)
                    generation += 1
                    fitness.append(fitness_value)
                except ValueError:
                    continue
  
    if generations and fitness:
        plt.figure(figsize=(10, 5))
        plt.plot(generations, fitness, marker='o', linestyle='-', color='blue')
        plt.title("Fitness do Melhor Indivíduo por Geração")
        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum dado válido encontrado no ficheiro.")
    

def main2():
    controller = Evolution()
    controller.run()
        

def main():
    controller = Evolution()
    if os.path.exists("melhores_individuos.txt") and have_20_individuals():
        population = get_weights_to_pop()
        population = [{'weights': np.array(ind), 'fitness': 0} for ind in population]
    else:
        population = inicialize_population_ann()

    generations_without_improvement = 0
    best_fitness_ever = float('-inf')
    fitness_history = []  
    SIGNIFICANT_IMPROVEMENT_THRESHOLD = 10  

    with open("melhores_individuos.txt", "a") as f:
        for generation in range(GENERATIONS):
            print(f"\nGeneration {generation+1}")

            for individual in population:
                individual['fitness'] = 0

                fitness1 = controller.runRobot(individual['weights'])
                fitness2 = controller.runRobot(individual['weights'])

                individual['fitness'] = (fitness1 + fitness2) / 2
                print(f"Fitness (mean of 2 runs): {individual['fitness']}")

            population_sorted = sorted_parents(population)
            best_individual = population_sorted[0]
            print(f"Best fitness: {best_individual['fitness']}")

            # Salvar o melhor fitness no histórico
            fitness_history.append(best_individual['fitness'])

            # Salvar o melhor indivíduo da geração no arquivo
            f.write(f"Fitness: {best_individual['fitness']}\n")
            f.write(f"Weights: {best_individual['weights'].tolist()}\n\n")
            f.flush()

            # Verificar se houve uma melhoria significativa
            if best_individual['fitness'] > best_fitness_ever + SIGNIFICANT_IMPROVEMENT_THRESHOLD:
                best_fitness_ever = best_individual['fitness']
                generations_without_improvement = 0
                print("Melhoria significativa detectada!")
            else:
                generations_without_improvement += 1
                print("Nenhuma melhoria significativa nesta geração.")

            # Recriar a população se não houver melhoria significativa por muitas gerações
            if generations_without_improvement >= NO_IMPROVEMENT_LIMIT:
                print(f"\nNenhuma melhoria significativa após {NO_IMPROVEMENT_LIMIT} gerações. Recriando população...\n")
                population = inicialize_population_ann()
                generations_without_improvement = 0
                continue

            # Gerar a próxima geração
            population = new_pop(population_sorted)

            # Salvar o gráfico a cada 100 gerações
            if (generation + 1) % 100 == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker='o', linestyle='-', color='blue')
                plt.title("Evolução do Fitness do Melhor Indivíduo")
                plt.xlabel("Geração")
                plt.ylabel("Fitness")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"fitness_evolution_gen_{generation+1}.png")
                plt.close()
                print(f"Gráfico salvo: fitness_evolution_gen_{generation+1}.png")

if __name__ == "__main__":
    #main()
    main2()
    #plot_graph()