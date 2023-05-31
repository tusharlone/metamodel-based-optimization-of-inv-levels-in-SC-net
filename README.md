# Meta-model-based Optimization of Inventory Levels in Supply Chains

The optimization task in Supply Chain is often time complex and done by using Simulation-based Optimization methods. The Supply Chain is simulated using any simulation tools, and then optimization is done using the simulation model as a black box. We tackle the inventory optimization problem in Supply Chain. We model the SC network using the *SimPy* library (Discrete Event Simulation) of Python and then apply meta-models to improve the efficacy and accuracy of the optimization process.

This repository contains a small Python project demonstrating the meta-model-based optimization process for inventory optimization in a Supply Chain network. It comprises four modules, 

1. model (the simulation model built using *SimPy*)
2. cost accuracy tradeoff (to estimate the computation budget of running the model)
3. design experimentation (setting the experiment and running the model to obtain the model data)
4. meta-model-based optimization (applying meta-model-based optimization on obtained data)

Each module has Python scripts and notebook files used for the optimization process. 