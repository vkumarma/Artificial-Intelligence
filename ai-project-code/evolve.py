from evolve_lib import *
import sys
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
import configparser

if __name__ == '__main__':
    options = sys.argv
    # Tk().withdraw()
    # filename = askopenfilename()  # filepath

    # Create a config parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')

    parameters = validate_parameters(config)
    FILENAME = parameters[0]
    POPULATION_SIZE = parameters[1]
    NUM_GENERATIONS = parameters[2]
    MUTATION_RATE = parameters[3]
    PERCENTAGE_GENERATED = parameters[4]

    print_parameters(FILENAME, POPULATION_SIZE, NUM_GENERATIONS,
                     MUTATION_RATE, PERCENTAGE_GENERATED)

    source_image = cv2.imread(FILENAME, 1)  # actual image without any processing
    population = generate_initial_population(POPULATION_SIZE, source_image)  # generates initial population
    X = int(PERCENTAGE_GENERATED * POPULATION_SIZE)  # number of chromosomes selected using selection
    Y = POPULATION_SIZE - X  # remaining generated using crossover
    gen = 0  # keeps track of generation
    highest_fitness_chromo = []

    print("\n")
    while gen <= NUM_GENERATIONS:
        # if gen != 0 and gen % 10 != 0:
        #     print("Generation: ", gen)
        for chromosome in population:
            score = calculate_fitness(chromosome, source_image)
            chromosome.fitness = score
            # cv2.imshow("Output_image", chromosome.image)
            # cv2.waitKey(0)
            # cv2.destroyWindow("Output_image")

        heap = ChromosomesHeap()  # heap initialization
        for solution in population:
            heap.add(solution)

        # if gen != 0 and gen % 10 == 0:
        print("Generation: ", gen, "max:", heap.max().fitness, "min:", heap.min().fitness, "average:",
              heap.average())

        if gen == NUM_GENERATIONS:
            highest_fitness_chromo.append(heap.min())
            break

        selected_chromosomes = elitist(X, heap)  # selection
        crossover_chromosomes = k_point(Y, selected_chromosomes, source_image)  # crossover
        new_chromosomes = selected_chromosomes + crossover_chromosomes
        population = mutation(MUTATION_RATE, new_chromosomes)  # mutation
        gen += 1

    print("\n")
    print("Generation: ", gen, " Highest Fitness Score: ", highest_fitness_chromo[0])
    cv2.imwrite('Output_image.jpg', highest_fitness_chromo[0].image)
    cv2.imshow("Output_image", highest_fitness_chromo[0].image)
    cv2.waitKey(0)
    cv2.destroyWindow("Output_image")
