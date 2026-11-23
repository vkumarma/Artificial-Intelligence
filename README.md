# Evolving Mona Lisa — Fresh Polygon Genetic Algorithm

This is a new implementation written from scratch. It does **not** reuse the previously supplied Python code.

## Result
The tested run creates a low-poly approximation using **2,240 filled triangles**. The face, hair, clothing, hands and landscape are represented by coloured polygons.

## Chromosome encoding
A chromosome is an ordered list of polygon genes. Every gene stores three (x, y) vertices and a BGR colour. The chromosome's polygon count is the length of its polygon list.

## Fitness function
Fitness is the sum of absolute pixel-by-pixel differences between the target image and the rendered chromosome. Lower fitness is better. Pixel values are converted to signed integers before subtraction to prevent unsigned integer overflow.

## Initial population
Each chromosome starts with a slightly different jittered triangular mesh. Polygon colours are sampled from the average target-image colour underneath each polygon, with random colour noise. This target-informed initialization is a deliberate design choice: completely random polygon positions and colours tend to create visual noise and require many more generations before a recognizable image appears.

## Genetic algorithm
- **Selection:** the best 20% of chromosomes are copied unchanged (elitism).
- **Crossover:** randomly selected parent pairs undergo one-point crossover, producing two children.
- **Mutation:** each polygon has a low probability of mutation. A mutation changes its colour or slightly moves one vertex.
- **Population:** population size remains constant.
- **Statistics:** minimum, maximum and average fitness are printed after every generation.

## Install and run

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python evolve_mona.py
```

For a machine without a graphical display:

```bash
python evolve_mona.py --headless
```

## Output files
- `Output_image.png` — best evolved polygon approximation.
- `fitness_history.csv` — minimum, maximum and average fitness for every generation.
- `monalisa.jpg` — target image.

## Configuration
Edit `config.ini` to change population size, generations, mutation rate and mesh resolution. More mesh rows and columns create more, smaller polygons and capture more detail, but increase runtime.

The supplied tested settings use 8 chromosomes, 35 generations, and a 40 x 28 mesh (two triangles per cell = 2,240 polygons).
