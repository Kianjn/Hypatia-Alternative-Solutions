
# Enhancements to Hypatia for Generating Near-Optimal Solutions (NOS)

This repository contains the work from my master's thesis, which focuses on enhancing the Hypatia energy modeling platform to generate near-optimal solutions (NOS). The modified version of Hypatia applies heuristic methods, including crossover and evolution techniques, to explore a diverse decision-space of energy system configurations.

## 🌟 Key Contributions
1. **Modifications to Hypatia**:
   - Integration of machine learning to iteratively adjust input parameters based on previous results.
   - Implementation of a heuristic NOS generation method, leveraging crossover and evolution techniques.

2. **Application to the Italian Energy Sector**:
   - Scenarios modeled for Italy's energy transition toward carbon neutrality.
   - Comparative analysis of results obtained through traditional optimization and NOS generation.

3. **Decision-Space Analysis**:
   - Visualization of diverse energy configurations within acceptable cost and emission thresholds.

## 🔗 Original Hypatia Platform
This project builds on the open-source [Hypatia platform](https://github.com/SESAM-Polimi/hypatia) developed by SESAM at Politecnico di Milano.

## 📂 Repository Structure
- `data/`: Input datasets and parameters specific to the Italian energy sector.
- `models/`: Python scripts, including:
  - The modified Hypatia framework.
  - NOS generation methods.
- `results/`: Outputs from simulations and decision-space visualizations.
- `docs/`: Thesis draft and other documentation.

## 🛠 Tools and Technologies

-	Hypatia: Open-source energy modeling platform.
- Gurobi: Optimization solver.
-	Python: For implementing heuristic methods and visualization.

## ✨ Methodology

The NOS generation approach leverages:
- Crossover: Combining characteristics of different solutions to create new configurations.
- Evolution: Iteratively refining solutions based on predefined fitness criteria.

## 🤝 Acknowledgments

- SESAM Lab at Politecnico di Milano for the original Hypatia platform.
-	Supervisors: Professor Emanuela Colombo, Francesco Cruz Torres

## 📖 References

- Original Hypatia: https://github.com/SESAM-Polimi/Hypatia-polimi
-	Gurobi Solver: https://www.gurobi.com
