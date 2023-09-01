# Solving the Unit Commitment problem using DWave's Quantum Annealers

This project was supported by [The University of Sydney Business School](https://www.sydney.edu.au/business/) Engaged Research initiative. 

This repository contains implementations and benchmarks of the Unit Commitment (UC) problem using DWave's Quantum Annealers. The UC problem is an NP-hard optimization problem in the field of power systems. It involves determining the optimal scheduling of power generation units to meet electricity demand while considering factors such as start-up costs, ramping constraints, and more. 

This repository aims to explore the application of Quantum Annealing to tackle this optimization problem, and to compare it and evaluate its performance against classical methods such as Mixed-Integer Linear Programming (MILP).

## Contents

1. [Files](#files)
2. [Usage](#usage)
3. [Data Source](#data-source)
4. [Contributing](#contributing)
5. [References](#references)

## Files

This repository contains the following main files:

1. [`simple_implementation.ipynb`](https://github.com/juanfrh7/uc-problem-annealing/blob/main/tests/simple_implementation.ipynb): This file presents a mathematical formulation and simple implementation of solving the UC problem using D-Wave's quantum annealer. It provides an example of how to formulate the UC problem as a constrained quadratic model and interface with the D-Wave QPU to find an approximate solution.

2. [`benchmarking.ipynb`](https://github.com/juanfrh7/uc-problem-annealing/blob/main/tests/benchmarking.ipynb): classical optimization techniques are employed to address the UC problem, with the goal of benchmarking quantum annealing against classical algorithms. The notebook undertakes several key analyses:
    - comparing and visualizing cost function outcomes to evaluate quantum's enhancements over classical,
    - calculating relative errors to quantify quantum accuracy,
    - assessing scalability through problem size incrementation,
    - exploring the effect of problem parameters on both quantum and classical performances,
    - measuring and visualizing the computational time for both approaches.

## Usage

To run the quantum annealing implementation, you need to have access to a D-Wave quantum annealer and the necessary credentials. Modify the parameters and constraints in [`simple_implementation.ipynb`](https://github.com/juanfrh7/uc-problem-annealing/blob/main/tests/simple_implementation.ipynb) as needed and run the script.

To benchmark quantum annealing against classical methods, use [`benchmarking.ipynb`](https://github.com/juanfrh7/uc-problem-annealing/blob/main/tests/benchmarking.ipynb). You might need to install specific optimization libraries for classical solvers.

## Data Source

The real-world datasets used in this project were obtained from the [power-system-optimization](https://github.com/east-winds/power-systems-optimization) repository. These datasets, originating from San Diego Gas and Electric (SDG&E) via the [PowerGenome](https://github.com/gschivley/PowerGenome) data platform, include:

- [Generators Data](https://github.com/juanfrh7/uc-problem-annealing/blob/main/data/Generators_data.csv): provides details on various generators, encompassing generator types, fuel types, minimum and maximum power generation capacities, etc.

- [Energy Demand Data](https://github.com/juanfrh7/uc-problem-annealing/blob/main/data/Demand.csv): hourly demand estimations for the year 2020, presented as net load at the transmission substation level after accounting for 600MW of behind-the-meter solar power.

- [Fuel Cost Data](https://github.com/juanfrh7/uc-problem-annealing/blob/main/data/Fuels_data.csv): includes estimated natural gas fuel costs.

- [Generator Variability Data](https://github.com/juanfrh7/uc-problem-annealing/blob/main/data/Generators_variability.csv): comprises variable generation capacity factors for PV, hydroelectric, and wind turbines.

## Contributing

Contributions to this repository are welcome! If you have ideas for improvements, additional benchmarks, or further optimizations, feel free to submit a pull request.

## References

[1] A. Bhardwaj, Vikram Kumar Kamboj, Vijay Kumar Shukla, B. Singh and P. Khurana, [Unit commitment in electrical power system-a literature review](https://ieeexplore.ieee.org/abstract/document/6230874), 2012 IEEE International Power Engineering and Optimization Conference Melaka, Malaysia, Melaka, Malaysia, 2012, pp. 275-280, doi: 10.1109/PEOCO.2012.6230874.

[2] L. Andrew, [Ising formulations of many NP problems](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005), Frontiers in Physics, 2, 2014, doi: 10.3389/fphy.2014.00005  

[3] S. Golestan, M.R. Habibi, S.Y. Mousazadeh Mousavi, J.M. Guerrero, J.C. Vasquez, [Quantum computation in power systems: An overview of recent advances](https://www.sciencedirect.com/science/article/pii/S2352484722025720), Energy Reports, Volume 9, 2023, doi: https://doi.org/10.1016/j.egyr.2022.11.185.  
