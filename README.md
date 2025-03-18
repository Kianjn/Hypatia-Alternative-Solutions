
# Enhancements to Hypatia for Generating Near-Optimal Solutions (NOS)

This repository contains the work from my master's thesis, which focuses on enhancing the Hypatia energy modeling platform to generate near-optimal solutions (NOS). The modified version of Hypatia applies heuristic methods, specifically crossover technique, to explore a diverse decision-space of energy system configurations.

## Near-Optimal Solutions

A solution that achieves an objective value (e.g., cost, emissions) very close to the best possible (optimal) outcome but has different configurations of technology choices, dispatch schedules, or infrastructure investments.

## 🌟 Key Contributions
1. **Modifications to Hypatia**:
   - Fully Automated Pipeline: Hypatia now Generates multiple near-optimal solutions with zero manual intervention.
   - Novel Method to Generate Alternatives: Hypatia uses the crossover technique of the heuristic method to effectively generate the decision space.
   - Quality & Quantity: Hypatia can generate a high number of solutions in a short time, and with a great diversity.
   - High User Input Compatibility: Users can define number of results, cost slack, and quality metrics.

2. **Application to the Italian Energy Sector**:
   - Comparative analysis of results obtained through traditional optimization and NOS generation.
   - Providing alternative pathways to achieve the same goal.
   - Offering insights about trade-offs between costs, emissions, and technological choices.
   - Visualization of diverse energy configurations within acceptable cost and emission thresholds.

## 🔗 Original Hypatia Platform
This project builds on the open-source [Hypatia platform](https://github.com/SESAM-Polimi/hypatia) developed by SESAM at Politecnico di Milano.

## 📂 Repository Structure
- `data/`: Input datasets and parameters specific to the Italian energy sector.
- `models/`: Python scripts, including:
  - The script added to the original framework, to enhance it. (enhancements.py)
- `results/`: Outputs from simulations and decision-space visualizations.
- `docs/`: Executive summary of the study, alongside the presentation.

## 🛠 Tools and Technologies

-	Hypatia: Open-source energy modeling platform.
-  Gurobi: Optimization solver.
-	Python: For implementing heuristic methods and visualization.

## ✨ Methodology

The NOS generation approach leverages:
- Crossover: Combining characteristics of different solutions to create new configurations.
- Evolution: Iteratively refining solutions based on predefined fitness criteria.

## 🤝 Acknowledgments

-  SESAM Lab at Politecnico di Milano for the original Hypatia platform.
-	Supervisors: Professor Emanuela Colombo, Francesco Cruz Torres

## 📖 References

- Original Hypatia: https://github.com/SESAM-Polimi/Hypatia-polimi
-	Gurobi Solver: https://www.gurobi.com
