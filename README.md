# Meta-model-based Optimization of Inventory Levels in Supply Chains

Simulation plays a key role in the design, analysis and optimization of Supply Chain systems. The Simulation-based Optimization (SBO) is often difficult owing to the large number of decision variables, the computational cost of performance estimation using multiple stochastic simulation runs and non-convex, black-box objective functions. Meta-models can assist in the optimization process, but the process of choosing the right meta-model type, the number of data points to build the meta-model and the right optimizer can be non-trivial and can significantly affect the results.  

This repository contains a Python project demonstrating the Meta-model-based Optimization process for inventory optimization in a Supply Chain network. It comprises four modules, 

1. model (the simulation model for a SC network built using *SimPy*)
2. cost accuracy tradeoff (to estimate the computation budget of running the model)
3. design experimentation (setting the experiment and running the model to obtain the model data)
4. meta-model-based optimization (applying meta-model-based optimization approach on obtained data)

Each module has Python scripts and notebook files used for the optimization process. 