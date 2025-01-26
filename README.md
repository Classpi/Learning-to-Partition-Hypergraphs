# Learning-to-Partition-Hypergraphs

Along with AI computing shining in scientific
discovery, its potential in the combinatorial opti
mization (CO) domain has also emerged in recent
years. Yet, existing unsupervised neural network
solvers are ill-suited for learning high-quality so
lutions on large-scale graph structures due to lim
ited ubiquitous computing infrastructures.
In this work, inspired by the remarkable achieve
ments of quantum algorithms in solving CO prob
lems, we adapt them to a wide range of multi
coloring problems on graph and hypergraphs (e.g.,
partitioning, clustering, coloring). To accommo
date the pairwise (or high-order) constraints on
graph (or hypergraph) structures, we first intro
duce novel one-hot encoded quadratic (or polyno
mial) unconstrained binary optimization model.
Leveraging GPUs, then we propose unified al
gorithms via matrix operation, ensuring efficient
computation and fast-training. On this basis, we
select graph partitioning as a solid instance and
propose PHP, a Physics-inspired approximate Hy
pergraph Partitioner. Performance comparison
with real-world datasets demonstrate PHP sur
passes classical heuristics.

## Project Structure

The project is organized as follows:

- `examples_GAP-Inspired/`: Examples for GAP-Inspired Loss Function
- `examples_standard-Form`: Examples for Standard-Form Loss Function
- `hgp/`: Main code area
- `scripts/` Control experiments using kahypar

## Run

1. install `miniconda` or `anaconda`.
2. in Project Folder,  run `conda env create -f environment.yaml` to create a environment.
3. run jupyter file using env created in step2