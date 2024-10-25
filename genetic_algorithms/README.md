Genetic Algorithm Parameters:
FILENAME: gen.txt
POPULATION_SIZE: 150
NUM_GENERATIONS: 100
SELECTION_TYPE: elitist
CROSSOVER_TYPE: uniform
MUTATION_RATE: 0.1
RATE_DECREASES: False
MUTATION_RATE_DELTA: 0.001
PERCENTAGE_GENERATED: 0.5


Generation:  10 max: 55.7 min: -46.53 average: -330.6
Generation:  20 max: 58.45 min: 53.25 average: -126.21
Generation:  30 max: 58.45 min: -5000 average: -524.42
Generation:  40 max: 64.66 min: -5000 average: -629.29
Generation:  50 max: 64.66 min: -58.45 average: -190.96
Generation:  60 max: 64.66 min: -5000 average: -387.48
Generation:  70 max: 64.66 min: -5000 average: -321.56
Generation:  80 max: 64.66 min: 58.37 average: -257.88
Generation:  90 max: 67.87 min: 64.66 average: -285.22
Generation:  100 max: 67.87 min: 67.87 average: -251.24


Generation:  100 Final Solution:  [-0.41  0.86 -0.71 -0.39  0.  ] Fitness Score:  67.87


Genetic Algorithm Parameters:
FILENAME: gen.txt
POPULATION_SIZE: 150
NUM_GENERATIONS: 100
SELECTION_TYPE: tournament
CROSSOVER_TYPE: uniform
MUTATION_RATE: 0.1
RATE_DECREASES: False
MUTATION_RATE_DELTA: 0.001
PERCENTAGE_GENERATED: 0.5


Generation:  10 max: 54.89 min: -5000 average: -204.46
Generation:  20 max: 54.89 min: 3.56 average: -235.79
Generation:  30 max: 56.46 min: 33.07 average: -539.08
Generation:  40 max: 58.45 min: -5000 average: -341.45
Generation:  50 max: 58.45 min: 51.73 average: -639.71
Generation:  60 max: 58.45 min: -5000 average: -272.92
Generation:  70 max: 58.45 min: -5000 average: -710.8
Generation:  80 max: 58.45 min: -34.7 average: -474.84
Generation:  90 max: 58.45 min: -15.2 average: -410.81
Generation:  100 max: 58.45 min: 37.24 average: -409.54


Generation:  100 Final Solution:  [-0.84  0.62 -2.28  2.92  1.  ] Fitness Score:  58.45

######################################################################################################################################################################################3#######################################################################################
The Genetic Algorithms seems to perform better with elitist selection and uniform crossover. Starting population must be high enough may be around 150 as that better solutions survive. 

[GENETIC_ALGORITHM]
FILENAME = gen.txt
POPULATION_SIZE = 150
NUM_GENERATIONS = 100
SELECTION_TYPE = elitist # or tournament
CROSSOVER_TYPE = uniform # or k-point
MUTATION_RATE = 0.1
RATE_DECREASES = False
MUTATION_RATE_DELTA = 0.001
PERCENTAGE_GENERATED = 0.5

# All the parameters are changeable. To change parameters set different values in config file.
# To run the code: python main.py
