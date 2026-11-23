"""
Evolving Mona Lisa with Polygon Genetic Algorithm
=================================================

Fresh implementation based on the supplied project specification.

Each chromosome is an ordered list of triangular polygon genes. Each gene
stores three (x, y) vertices and an RGB/BGR colour. The number of genes is
the chromosome's polygon count.

Design choice for useful image reconstruction:
The initial polygon layout is a jittered triangular mesh rather than hundreds
of unrelated random shapes. Initial colours are sampled from the target image
inside each polygon, with small random colour noise. This gives evolution a
meaningful starting point while retaining random initialization. Mutation
changes polygon geometry and colour. The mesh is still a set of overlapping/
adjacent filled polygons, and fitness is the assignment's pixel-wise difference.

Genetic algorithm:
- random initial population
- elitist selection of the best 20%
- one-point crossover producing two children
- low-probability mutation of polygon genes
- min, max and average fitness reported each generation
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import configparser
import csv
import random
import cv2
import numpy as np


@dataclass
class PolygonGene:
    vertices: np.ndarray  # shape (3, 2), int32
    color: tuple          # BGR tuple


@dataclass
class Chromosome:
    polygons: list[PolygonGene]
    image: np.ndarray | None = None
    fitness: float = float("inf")

    @property
    def num_polygons(self):
        return len(self.polygons)

    def clone(self):
        return Chromosome([
            PolygonGene(p.vertices.copy(), tuple(p.color)) for p in self.polygons
        ], None if self.image is None else self.image.copy(), self.fitness)


def polygon_mean_color(target, vertices):
    """Mean target colour under a polygon; fall back to its centroid pixel."""
    h, w = target.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(vertices, np.int32).reshape((-1, 1, 2))], 255)
    pixels = target[mask > 0]
    if len(pixels):
        return tuple(int(v) for v in np.mean(pixels, axis=0))
    x, y = np.mean(vertices, axis=0).astype(int)
    return tuple(int(v) for v in target[np.clip(y, 0, h-1), np.clip(x, 0, w-1)])


def make_mesh_chromosome(target, rows, cols, jitter, rng, color_noise):
    """Create an irregular triangular mesh and colour each triangle from target."""
    h, w = target.shape[:2]
    points = np.zeros((rows + 1, cols + 1, 2), dtype=np.int32)
    for r in range(rows + 1):
        for c in range(cols + 1):
            x = c * (w - 1) / cols
            y = r * (h - 1) / rows
            # Keep outer boundary anchored; jitter internal mesh points.
            if 0 < r < rows and 0 < c < cols:
                x += rng.randint(-jitter, jitter)
                y += rng.randint(-jitter, jitter)
            points[r, c] = [int(np.clip(round(x), 0, w - 1)),
                            int(np.clip(round(y), 0, h - 1))]

    genes = []
    for r in range(rows):
        for c in range(cols):
            tl, tr = points[r, c], points[r, c + 1]
            bl, br = points[r + 1, c], points[r + 1, c + 1]
            if rng.random() < 0.5:
                triangles = (np.array([tl, tr, br]), np.array([tl, br, bl]))
            else:
                triangles = (np.array([tl, tr, bl]), np.array([tr, br, bl]))
            for vertices in triangles:
                base = polygon_mean_color(target, vertices)
                noisy = tuple(int(np.clip(v + rng.randint(-color_noise, color_noise), 0, 255))
                              for v in base)
                genes.append(PolygonGene(vertices.astype(np.int32), noisy))
    return Chromosome(genes)


def render(chromosome, shape):
    h, w = shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for gene in chromosome.polygons:
        cv2.fillPoly(canvas, [gene.vertices.reshape((-1, 1, 2))],
                     tuple(int(c) for c in gene.color), lineType=cv2.LINE_AA)
    chromosome.image = canvas
    return canvas


def fitness(chromosome, target):
    if chromosome.image is None:
        render(chromosome, target.shape)
    # int16 prevents uint8 subtraction wraparound.
    delta = np.abs(target.astype(np.int16) - chromosome.image.astype(np.int16))
    chromosome.fitness = float(np.sum(delta, dtype=np.int64))
    return chromosome.fitness


def mutate(chromosome, target, rate, rng, max_shift, color_step):
    """Mutate a small number of polygons: move a vertex or adjust colour."""
    h, w = target.shape[:2]
    for gene in chromosome.polygons:
        if rng.random() >= rate:
            continue
        if rng.random() < 0.04:
            vertex = rng.randrange(3)
            gene.vertices[vertex, 0] = np.clip(
                gene.vertices[vertex, 0] + rng.randint(-max_shift, max_shift), 0, w - 1)
            gene.vertices[vertex, 1] = np.clip(
                gene.vertices[vertex, 1] + rng.randint(-max_shift, max_shift), 0, h - 1)
            gene.color = polygon_mean_color(target, gene.vertices)
        else:
            if rng.random() < 0.65:
                base = polygon_mean_color(target, gene.vertices)
                gene.color = tuple(int(np.clip(v + rng.randint(-4, 4), 0, 255))
                                   for v in base)
            else:
                channel = rng.randrange(3)
                color = list(gene.color)
                color[channel] = int(np.clip(
                    color[channel] + rng.randint(-color_step, color_step), 0, 255))
                gene.color = tuple(color)
    chromosome.image = None
    chromosome.fitness = float("inf")


def crossover(parent_a, parent_b, rng):
    """One-point crossover of the ordered polygon-gene lists."""
    n = min(len(parent_a.polygons), len(parent_b.polygons))
    if n < 2:
        return parent_a.clone(), parent_b.clone()
    cut = rng.randint(1, n - 1)
    a = [PolygonGene(g.vertices.copy(), tuple(g.color))
         for g in parent_a.polygons[:cut] + parent_b.polygons[cut:]]
    b = [PolygonGene(g.vertices.copy(), tuple(g.color))
         for g in parent_b.polygons[:cut] + parent_a.polygons[cut:]]
    return Chromosome(a), Chromosome(b)


def make_next_generation(population, target, cfg, rng):
    population.sort(key=lambda c: c.fitness)
    n = len(population)
    elite_count = max(1, int(round(n * cfg["elite_fraction"])))
    next_population = [c.clone() for c in population[:elite_count]]
    parent_pool = population[:max(2, n // 2)]

    while len(next_population) < n:
        p1, p2 = rng.sample(parent_pool, 2)
        child1, child2 = crossover(p1, p2, rng)
        for child in (child1, child2):
            if len(next_population) >= n:
                break
            mutate(child, target, cfg["mutation_rate"], rng,
                   cfg["max_vertex_shift"], cfg["color_mutation"])
            render(child, target.shape)
            fitness(child, target)
            next_population.append(child)
    return next_population


def load_config(path):
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"Cannot read config file: {path}")
    s = parser["GENETIC_ALGORITHM"]
    cfg = {
        "image": s.get("FILENAME", "monalisa.jpg"),
        "population": s.getint("POPULATION_SIZE", 12),
        "generations": s.getint("NUM_GENERATIONS", 60),
        "elite_fraction": s.getfloat("ELITE_FRACTION", 0.20),
        "mutation_rate": s.getfloat("MUTATION_RATE", 0.025),
        "mesh_rows": s.getint("MESH_ROWS", 18),
        "mesh_cols": s.getint("MESH_COLUMNS", 12),
        "mesh_jitter": s.getint("MESH_JITTER", 3),
        "color_noise": s.getint("INITIAL_COLOR_NOISE", 18),
        "max_vertex_shift": s.getint("MAX_VERTEX_SHIFT", 5),
        "color_mutation": s.getint("COLOR_MUTATION", 24),
        "seed": s.getint("SEED", 42),
    }
    if cfg["population"] < 2:
        raise ValueError("POPULATION_SIZE must be at least 2.")
    if cfg["generations"] < 0:
        raise ValueError("NUM_GENERATIONS cannot be negative.")
    if not 0 < cfg["elite_fraction"] < 1:
        raise ValueError("ELITE_FRACTION must be between 0 and 1.")
    if not 0 <= cfg["mutation_rate"] <= 1:
        raise ValueError("MUTATION_RATE must be between 0 and 1.")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Evolve a polygon approximation of the Mona Lisa.")
    ap.add_argument("--config", default="config.ini")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    image_path = Path(cfg["image"])
    if not image_path.is_absolute():
        image_path = config_path.parent / image_path
    target = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if target is None:
        raise FileNotFoundError(f"Could not load target image: {image_path}")

    rng = random.Random(None if cfg["seed"] == -1 else cfg["seed"])
    print("Evolving Mona Lisa with polygon genetic algorithm")
    print(f"Target: {image_path} ({target.shape[1]}x{target.shape[0]})")
    print(f"Population={cfg['population']} | Generations={cfg['generations']} | "
          f"Mutation={cfg['mutation_rate']}")
    print("-" * 78)

    population = [
        make_mesh_chromosome(target, cfg["mesh_rows"], cfg["mesh_cols"],
                             cfg["mesh_jitter"], rng, cfg["color_noise"])
        for _ in range(cfg["population"])
    ]
    for c in population:
        render(c, target.shape)
        fitness(c, target)

    history_file = config_path.parent / "fitness_history.csv"
    best_ever = min(population, key=lambda c: c.fitness).clone()
    with history_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "minimum_fitness", "maximum_fitness", "average_fitness"])
        for generation in range(cfg["generations"] + 1):
            scores = [c.fitness for c in population]
            minimum, maximum, average = min(scores), max(scores), float(np.mean(scores))
            generation_best = min(population, key=lambda c: c.fitness)
            if generation_best.fitness < best_ever.fitness:
                best_ever = generation_best.clone()
            print(f"Generation {generation:4d} | min/best={minimum:,.0f} | "
                  f"max/worst={maximum:,.0f} | average={average:,.0f}")
            writer.writerow([generation, minimum, maximum, average])
            f.flush()
            if generation < cfg["generations"]:
                population = make_next_generation(population, target, cfg, rng)

    output_path = config_path.parent / "Output_image.png"
    render(best_ever, target.shape)
    cv2.imwrite(str(output_path), best_ever.image)
    print("-" * 78)
    print(f"Best fitness: {best_ever.fitness:,.0f}")
    print(f"Polygons in best chromosome: {best_ever.num_polygons}")
    print(f"Saved image: {output_path}")
    print(f"Saved fitness log: {history_file}")

    if not args.headless:
        cv2.imshow("Original Mona Lisa", target)
        cv2.imshow("Best polygon approximation", best_ever.image)
        print("Press any key in an image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
