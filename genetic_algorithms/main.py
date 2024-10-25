import configparser
import heapq
import os
import copy
import random
import numpy as np




# Create a config parser
config = configparser.ConfigParser()

# Read the configuration file
config.read('config.ini')


# Access and validate the parameters
def validate_parameters(config):
    FILENAME = config.get('GENETIC_ALGORITHM', 'FILENAME')
    POPULATION_SIZE = config.getint('GENETIC_ALGORITHM', 'POPULATION_SIZE')
    NUM_GENERATIONS = config.getint('GENETIC_ALGORITHM', 'NUM_GENERATIONS')
    SELECTION_TYPE = config.get('GENETIC_ALGORITHM', 'SELECTION_TYPE')
    CROSSOVER_TYPE = config.get('GENETIC_ALGORITHM', 'CROSSOVER_TYPE')
    MUTATION_RATE = config.getfloat('GENETIC_ALGORITHM', 'MUTATION_RATE')
    RATE_DECREASES = config.getboolean('GENETIC_ALGORITHM', 'RATE_DECREASES')
    MUTATION_RATE_DELTA = config.getfloat('GENETIC_ALGORITHM', 'MUTATION_RATE_DELTA')
    PERCENTAGE_GENERATED = config.getfloat('GENETIC_ALGORITHM', 'PERCENTAGE_GENERATED')

    # Validation
    if POPULATION_SIZE <= 0:
        raise ValueError("POPULATION_SIZE must be a positive integer.")
    if NUM_GENERATIONS <= 0:
        raise ValueError("NUM_GENERATIONS must be a positive integer.")

    valid_selection_types = ['elitist', 'tournament']
    if SELECTION_TYPE not in valid_selection_types:
        raise ValueError(f"SELECTION_TYPE must be one of: {valid_selection_types}.")

    valid_crossover_types = ['uniform', 'k-point']
    if CROSSOVER_TYPE not in valid_crossover_types:
        raise ValueError(f"CROSSOVER_TYPE must be one of: {valid_crossover_types}.")

    if not (0 <= MUTATION_RATE <= 1):
        raise ValueError("MUTATION_RATE must be between 0 and 1.")

    if RATE_DECREASES and (MUTATION_RATE_DELTA <= 0):
        raise ValueError("MUTATION_RATE_DELTA must be positive if RATE_DECREASES is True.")

    if not (0 <= PERCENTAGE_GENERATED <= 1):
        raise ValueError("PERCENTAGE_GENERATED must be between 0 and 1.")

    return (FILENAME, POPULATION_SIZE, NUM_GENERATIONS, SELECTION_TYPE,
            CROSSOVER_TYPE, MUTATION_RATE, RATE_DECREASES,
            MUTATION_RATE_DELTA, PERCENTAGE_GENERATED)


MEAN = 0
STD_DEV = 1.15
SEED_NUMBER = 42
np.random.seed(SEED_NUMBER)
random.seed(SEED_NUMBER)

class Chromosomes_heap:
    def __init__(self):
        self.heap = []
        heapq.heapify(self.heap)

    def add(self, chromo):
        heapq.heappush(self.heap, chromo)

    def remove(self):  # removes or pops element with the highest fitness score
        if len(self.heap) != 0:
            return heapq.heappop(self.heap)

    def at_index(self, i):  # chromosome at particular index
        return self.heap[i]

    def size(self):  # number of chromosomes
        return len(self.heap)

    def min(self):  # returns chromosome with minimum fitness score
        return self.heap[-1]

    def max(self):  # returns chromosome with maximum fitness score
        return self.heap[0]

    def average(self):  # returns average fitness score
        return round(sum([chromo.fitness for chromo in self.heap]) / self.size(), 2)

class Chromosome:
    def __init__(self, chromo):

        if chromo[0] > chromo[1]:
            chromo[0], chromo[1] = chromo[1], chromo[0]
        if chromo[2] > chromo[3]:
            chromo[2], chromo[3] = chromo[3], chromo[2]

        self.encoding = chromo
        self.fitness = -5000

    def __lt__(self, other):  # max heap
        return self.fitness > other.fitness

    def __len__(self):
        return len(self.encoding)


def process_data(file_name):
    data = []
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, 'r') as file:
        for line in file:
            data.append([float(elem) for elem in line.split()])

    return data


def fitness_score(chromo, data):  # returns fitness score for a particular chromosome or scores how good a particular solution is
    fitness, match = 0, False
    for row in data:
        if chromo.encoding[0] < row[0] < chromo.encoding[1]:
            if chromo.encoding[2] < row[1] < chromo.encoding[3]:
                match = True
                if chromo.encoding[-1] == 1:  # bought the stock
                    fitness += row[-1]
                else:  # shorted the stock
                    fitness += -1 * row[-1]

    return match, round(fitness, 2)


def elitist(x, chromosomes):  # select top x chromosomes
    selected = []
    for i in range(x):
        selected.append(chromosomes.remove())

    return selected

def tournament_selection(x, chromosomes):  # select two chromos at random and compete or choose the one with max fitness
    selected, i = [], 0
    while i < x:
        index1 = random.randint(0, chromosomes.size()-1)
        index2 = random.randint(0, chromosomes.size()-1)

        if chromosomes.at_index(index1).fitness > chromosomes.at_index(index2).fitness:  # add chromosome with max fitness
            selected.append(chromosomes.at_index(index1))
        else:
            selected.append(chromosomes.at_index(index2))

        i += 1
    return selected

def selection(algo_type,  x, chromosomes):  # selection type, X number of chromosomes cloned into the next
    # generation, chromosomes -  heap

    if algo_type == 'elitist':
        return elitist(x, chromosomes)

    elif algo_type == 'tournament':
        return tournament_selection(x, chromosomes)

def uniform(y, chromosomes):  # selects 2 chromosomes at random and iterate through each gene, randomly chooses either of the chromosomes gene
    cross_over = []
    k = 0
    while k < y:
        index1 = random.randint(0, len(chromosomes)-1)
        index2 = random.randint(0, len(chromosomes)-1)
        encoding = np.zeros(5)

        chromo1 = chromosomes[index1]  # randomly choose two chromosomes
        chromo2 = chromosomes[index2]

        for i in range(len(encoding)):
            p = random.randint(0, 1)  # 0 or 1
            if p == 0:
                encoding[i] = chromo1.encoding[i]
            else:
                encoding[i] = chromo2.encoding[i]

        ch = Chromosome(encoding)
        cross_over.append(ch)
        k += 1

    return cross_over

def k_point(y, chromosomes):  # randomly chooses two chromosomes and selects first half from one chromosome and another half from other chromosome in a single-point crossover
    cross_over, k = [], 0
    while k < y:
        index1 = random.randint(0, len(chromosomes) - 1)
        index2 = random.randint(0, len(chromosomes) - 1)
        encoding = np.zeros(5)

        chromo1 = chromosomes[index1]  # randomly choose two chromosomes
        chromo2 = chromosomes[index2]

        for i in range(0, 2):
            encoding[i] = chromo1.encoding[i]

        for j in range(2, len(encoding)):
            encoding[j] = chromo2.encoding[j]

        ch = Chromosome(encoding)
        cross_over.append(ch)
        k += 1

    return cross_over

def crossover(algo_type, y, chromosomes):  # algo_type, y - num of chromosomes on which crossover is applied, chromosomes - from selection

    if algo_type == 'uniform':
        return uniform(y, chromosomes)

    elif algo_type == 'k-point':
        return k_point(y, chromosomes)


def mutation(rate, chromosomes):  # mutation is applied on new chromosomes generated from selection and crossover
    for chromo in chromosomes:
        encoding = chromo.encoding
        for i in range(len(encoding)):
            if random.random() < rate:
                if i == len(encoding) - 1:  # last gene of chromosome must either be 0 or 1
                    encoding[i] = random.randint(0, 1)
                else:
                    encoding[i] = round(np.random.normal(MEAN, STD_DEV), 2)  #

        if encoding[0] > encoding[1]:
            encoding[0], encoding[1] = encoding[1], encoding[0]
        if encoding[2] > encoding[3]:
            encoding[2], encoding[3] = encoding[3], encoding[2]
        chromo.encoding = encoding

    return chromosomes

def generate_initial_population(size):
    chromosomes = []
    for i in range(size):
        random_array = np.random.normal(MEAN, STD_DEV, 5)
        random_array = np.round(random_array, 2)
        random_array[-1] = random.randint(0, 1)
        ch = Chromosome(random_array)
        chromosomes.append(ch)

    return chromosomes

def print_parameters(filename, population_size, num_generations, selection_type,
                     crossover_type, mutation_rate, rate_decreases,
                     mutation_rate_delta, percentage_generated):
    print("Genetic Algorithm Parameters:")
    print(f"FILENAME: {filename}")
    print(f"POPULATION_SIZE: {population_size}")
    print(f"NUM_GENERATIONS: {num_generations}")
    print(f"SELECTION_TYPE: {selection_type}")
    print(f"CROSSOVER_TYPE: {crossover_type}")
    print(f"MUTATION_RATE: {mutation_rate}")
    print(f"RATE_DECREASES: {rate_decreases}")
    print(f"MUTATION_RATE_DELTA: {mutation_rate_delta}")
    print(f"PERCENTAGE_GENERATED: {percentage_generated}")


parameters = validate_parameters(config)
# Set constants
FILENAME = parameters[0]
POPULATION_SIZE = parameters[1]
NUM_GENERATIONS = parameters[2]
SELECTION_TYPE = parameters[3]
CROSSOVER_TYPE = parameters[4]
MUTATION_RATE = parameters[5]
RATE_DECREASES = parameters[6]
MUTATION_RATE_DELTA = parameters[7]
PERCENTAGE_GENERATED = parameters[8]

# Print the constants
print_parameters(FILENAME, POPULATION_SIZE, NUM_GENERATIONS, SELECTION_TYPE,
                 CROSSOVER_TYPE, MUTATION_RATE, RATE_DECREASES,
                 MUTATION_RATE_DELTA, PERCENTAGE_GENERATED)


data = process_data(FILENAME)
population = generate_initial_population(POPULATION_SIZE)  # generates initial population
X = int(PERCENTAGE_GENERATED * POPULATION_SIZE)  # number of chromosomes selected using selection
Y = POPULATION_SIZE - X  # remaining generated using crossover
gen = 0  # keeps track of generation
highest_fitness_chromo = []
print("\n")
while gen <= NUM_GENERATIONS:
    # if gen != 0 and gen % 10 != 0:
    #     print("Generation: ", gen)
    for chromosome in population:
        matched, score = fitness_score(chromosome, data)
        if matched:
            chromosome.fitness = score

    heap = Chromosomes_heap()  # adding chromosomes to the heap
    for solution in population:
        heap.add(solution)

    if gen != 0 and gen % 10 == 0:
        print("Generation: ", gen, "max:", heap.max().fitness, "min:", heap.min().fitness, "average:", heap.average())

    if gen == NUM_GENERATIONS:
        highest_fitness_chromo.append(heap.max())
        break

    selected_chromosomes = selection(SELECTION_TYPE, X, heap)
    crossover_chromosomes = crossover(CROSSOVER_TYPE, Y, selected_chromosomes)
    new_chromosomes = selected_chromosomes + crossover_chromosomes
    population = mutation(MUTATION_RATE, new_chromosomes)

    if RATE_DECREASES:  # if this flag is true mutation rate will decrease by some value
        if MUTATION_RATE - MUTATION_RATE_DELTA > 0:
            MUTATION_RATE -= MUTATION_RATE_DELTA
    gen += 1


print("\n")
print("Generation: ", gen, "Final Solution: ", highest_fitness_chromo[0].encoding, "Fitness Score: ", highest_fitness_chromo[0].fitness)