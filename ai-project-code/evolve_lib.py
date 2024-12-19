import numpy as np
import heapq
import cv2
import random

SEED_NUMBER = 42
np.random.seed(SEED_NUMBER)
random.seed(SEED_NUMBER)


class ChromosomesHeap:
    def __init__(self):
        self.heap = []
        heapq.heapify(self.heap)

    def add(self, chromo):
        heapq.heappush(self.heap, chromo)

    def remove(self):
        if self.size() > 0:
            return heapq.heappop(self.heap)
        raise IndexError("Heap is empty")

    def at_index(self, i):
        if 0 <= i < self.size():
            return self.heap[i]
        raise IndexError("Index out of range")

    def size(self):
        return len(self.heap)

    def min(self):
        if self.size() > 0:
            return self.heap[0]
        raise IndexError("Heap is empty")

    def max(self):
        if self.size() > 0:
            return max(self.heap, key=lambda chromo: chromo.fitness)
        raise IndexError("Heap is empty")

    def average(self):
        if self.size() > 0:
            return round(sum(chromo.fitness for chromo in self.heap) / self.size(), 2)
        raise ValueError("Heap is empty")


class Polygon:
    def __init__(self, vertices, color):
        self.vertices = vertices
        self.color = color

    def __repr__(self):
        return f"Polygon(vertices={self.vertices}, color={self.color})"


class Chromosome:
    def __init__(self, image, num_polygons, fitness, polygons):
        self.image = image  # encoding
        self.num_polygons = num_polygons  # genes
        self.fitness = fitness  # fitness score
        self.polygons = polygons  # stores objects of polygons with vertices and color

    def __lt__(self, other):  # min heap - choose chromosomes / individuals with minimum score or minimum difference
        # between target image and chromosome image
        return self.fitness < other.fitness

    def __repr__(self):
        return f"Chromosome(fitness={self.fitness})"


def validate_parameters(config):
    FILENAME = config.get('GENETIC_ALGORITHM', 'FILENAME')
    POPULATION_SIZE = config.getint('GENETIC_ALGORITHM', 'POPULATION_SIZE')
    NUM_GENERATIONS = config.getint('GENETIC_ALGORITHM', 'NUM_GENERATIONS')
    MUTATION_RATE = config.getfloat('GENETIC_ALGORITHM', 'MUTATION_RATE')
    PERCENTAGE_GENERATED = config.getfloat('GENETIC_ALGORITHM', 'PERCENTAGE_GENERATED')

    # Validation
    if POPULATION_SIZE <= 0:
        raise ValueError("POPULATION_SIZE must be a positive integer.")
    if NUM_GENERATIONS <= 0:
        raise ValueError("NUM_GENERATIONS must be a positive integer.")

    if not (0 <= MUTATION_RATE <= 1):
        raise ValueError("MUTATION_RATE must be between 0 and 1.")

    if not (0 <= PERCENTAGE_GENERATED <= 1):
        raise ValueError("PERCENTAGE_GENERATED must be between 0 and 1.")

    return FILENAME, POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, PERCENTAGE_GENERATED


def draw(image, num_sides, radius, color):  # draws polygons on the image and validates
    height, width, _ = image.shape
    center_x, center_y = random.randint(0, width - 1), random.randint(0, height - 1)

    vertices = []
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides  # Angle in radians
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        vertices.append([x, y])

    # Calculate the minimum x and y values
    min_x = min([location[0] for location in vertices])
    min_y = min([location[1] for location in vertices])

    # Shift the polygon only if it goes out of bounds
    if min_x < 0:
        for i in range(0, len(vertices)):
            pair = vertices[i]
            pair[0] -= min_x

    if min_y < 0:
        for i in range(0, len(vertices)):
            pair = vertices[i]
            pair[1] -= min_y

    # Ensure the polygon doesn't exceed the maximum bounds
    max_x = max([location[0] for location in vertices])
    max_y = max([location[1] for location in vertices])

    if max_x > width:
        for i in range(0, len(vertices)):
            pair = vertices[i]
            pair[0] -= (max_x - width)

    if max_y > height:
        for i in range(0, len(vertices)):
            pair = vertices[i]
            pair[1] -= (max_y - height)

    polygon = np.array(vertices, np.int32).reshape((-1, 1, 2))
    image = image.copy()
    image = cv2.fillPoly(image, [polygon], color)
    polygon_object = Polygon(vertices, color)

    return image, polygon_object


def generate_initial_population(size, source):
    chromosomes = []  # list of chromosomes or initial population
    height, width, channels = source.shape
    for i in range(size):  # size here refers to population size
        polygons = []
        image = np.full(shape=(height, width, channels), fill_value=255, dtype=np.uint8)  # encoding
        num_polygons = 400  # number of polygons

        for j in range(num_polygons):
            num_sides = random.randint(3, 12)  # different sides
            radius = random.randint(1, 30)  # polygon size

            color = (  # different colors
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            image, poly_object = draw(image, num_sides, radius, color)  # drawing polygon on the image
            polygons.append(poly_object)

        chromosome = Chromosome(image, num_polygons, float('inf'), polygons)
        chromosomes.append(chromosome)

    return chromosomes


def elitist(x, chromosomes):  # select top x chromosomes
    if x > chromosomes.size():
        raise ValueError("Number of elite chromosomes requested exceeds population size.")

    selected = []
    for i in range(x):
        selected.append(chromosomes.remove())

    return selected


def calculate_fitness(chromosome, source):  # how fit the chromosome is
    score = np.sum(abs(source - chromosome.image))  # normalizing score as well
    return score


def k_point(y, chromosomes, source):  # mixing up shapes to create new children
    cross_over, k = [], 0
    height, width, channels = source.shape

    while k < (y // 2):
        # Initialize blank images for two children
        image1 = np.full(shape=(height, width, channels), fill_value=255, dtype=np.uint8)
        image2 = np.full(shape=(height, width, channels), fill_value=255, dtype=np.uint8)

        # Select two distinct parents
        index1 = random.randint(0, len(chromosomes) - 1)
        index2 = random.randint(0, len(chromosomes) - 1)
        while index1 == index2:
            index2 = random.randint(0, len(chromosomes) - 1)

        parent_1 = chromosomes[index1]
        parent_2 = chromosomes[index2]

        # Split polygons from both parents
        mid_point1 = len(parent_1.polygons) // 2
        mid_point2 = len(parent_2.polygons) // 2

        child1_polygons = parent_1.polygons[:mid_point1] + parent_2.polygons[mid_point2:]
        random.shuffle(child1_polygons)
        child2_polygons = parent_2.polygons[:mid_point2] + parent_1.polygons[mid_point1:]
        random.shuffle(child2_polygons)

        # Draw polygons on the images for child 1
        for polygon in child1_polygons:
            vertices = np.array(polygon.vertices, np.int32).reshape((-1, 1, 2))  # Convert to NumPy array
            image1 = cv2.fillPoly(image1, [vertices], color=polygon.color)

        # Draw polygons on the images for child 2
        for polygon in child2_polygons:
            vertices = np.array(polygon.vertices, np.int32).reshape((-1, 1, 2))  # Convert to NumPy array
            image2 = cv2.fillPoly(image2, [vertices], color=polygon.color)

        # Create child chromosomes
        child1 = Chromosome(image1, len(child1_polygons), float('inf'), child1_polygons)
        child2 = Chromosome(image2, len(child2_polygons), float('inf'), child2_polygons)

        # Add children to the new population
        cross_over.append(child1)
        cross_over.append(child2)
        k += 1

    return cross_over


def uniform(y, chromosomes, source):  # randomly chooses two chromosomes and selects first half from one chromosome and
    # another half from other chromosome in a single-point crossover
    cross_over, k = [], 0
    height, width, channels = source.shape
    while k < (y // 2):
        image1 = np.full(shape=(height, width, channels), fill_value=255, dtype=np.uint8)
        image2 = np.full(shape=(height, width, channels), fill_value=255, dtype=np.uint8)

        index1 = random.randint(0, len(chromosomes) - 1)
        index2 = random.randint(0, len(chromosomes) - 1)

        while index1 == index2:
            index2 = random.randint(0, len(chromosomes) - 1)

        parent_1 = chromosomes[index1]  # randomly choose two chromosomes
        parent_2 = chromosomes[index2]

        all_polygons = parent_1.polygons + parent_2.polygons  # Combine polygons from both parents
        random.shuffle(all_polygons)  # Shuffle the polygons to introduce diversity

        # split_point = len(all_polygons) // 2  # Ensure equal or nearly equal division
        child1_polygons = []
        child2_polygons = []

        for i in range(len(all_polygons) - 1):
            p = random.randint(0, 1)  # 0 or 1
            if p == 0:
                child1_polygons.append(all_polygons[i])
            else:
                child2_polygons.append(all_polygons[i])

        for polygon in child1_polygons:
            vertices = np.array(polygon.vertices, np.int32).reshape((-1, 1, 2))  # Convert to NumPy array
            image1 = cv2.fillPoly(image1, [vertices], color=polygon.color)

        for polygon in child2_polygons:
            vertices = np.array(polygon.vertices, np.int32).reshape((-1, 1, 2))  # Convert to NumPy array
            image2 = cv2.fillPoly(image2, [vertices], color=polygon.color)

        child1 = Chromosome(image1, len(child1_polygons), float('inf'), child1_polygons)
        child2 = Chromosome(image2, len(child2_polygons), float('inf'), child2_polygons)
        cross_over.append(child1)
        cross_over.append(child2)
        k += 1

    return cross_over


def mutation(rate, chromosomes):
    for chromo in chromosomes:
        if not chromo.polygons:
            continue  # Skip chromosomes with no polygons

        image = chromo.image.copy()
        height, width, channels = image.shape

        if random.random() < rate:
            # Randomly select two different polygons to swap
            index1, index2 = random.randint(0, len(chromo.polygons) - 1), random.randint(0, len(chromo.polygons) - 1)
            polygon1, polygon2 = chromo.polygons[index1], chromo.polygons[index2]
            # Swap their vertices
            polygon1.vertices, polygon2.vertices = polygon2.vertices, polygon1.vertices

            # Remove both polygons from the image
            vertices1 = np.array(polygon1.vertices, dtype=np.int32).reshape((-1, 1, 2))
            vertices2 = np.array(polygon2.vertices, dtype=np.int32).reshape((-1, 1, 2))
            image = cv2.fillPoly(image, [vertices1], (255, 255, 255))
            image = cv2.fillPoly(image, [vertices2], (255, 255, 255))

            # Redraw polygons at swapped locations
            image = cv2.fillPoly(image, [np.array(polygon1.vertices, np.int32).reshape((-1, 1, 2))], polygon1.color)
            image = cv2.fillPoly(image, [np.array(polygon2.vertices, np.int32).reshape((-1, 1, 2))], polygon2.color)

        # Update the chromosome's image
        chromo.image = image

    return chromosomes


def print_parameters(filename, population_size, num_generations, mutation_rate, percentage_generated):
    print("Genetic Algorithm Parameters:")
    print(f"FILENAME: {filename}")
    print(f"POPULATION_SIZE: {population_size}")
    print(f"NUM_GENERATIONS: {num_generations}")
    print(f"MUTATION_RATE: {mutation_rate}")
    print(f"PERCENTAGE_GENERATED: {percentage_generated}")
