---
sidebar_position: 7
---


Some information are from [jkoendev/optimal-control-literature-software: List of literature and software for optimal control and numerical optimization.](https://github.com/jkoendev/optimal-control-literature-software)

# Literature and list of software packages for optimal control

The list includes resources to the following topics: Automatic/algorithmic differentiation, optimal control, model-predictive control (MPC), numerical optimization, modeling for control.  The list will be updated regularly, create a pull request if you'd like to contribute.

## Literature

### Lectures

* Lecture notes: Numerical Optimal Control by Prof. Moritz Diehl [[course](https://www.syscop.de/teaching/ss2017/numerical-optimal-control)] [[pdf](https://www.syscop.de/files/2017ss/NOC/script/book-NOCSE.pdf)]
* Tutorial series by Metthew Kelly, [[web](http://www.matthewpeterkelly.com/tutorials/index.html)]
* Liberzon, Daniel. Calculus of variations and optimal control theory: a concise introduction. Princeton University Press, 2011. [[pre-print](http://liberzon.csl.illinois.edu/teaching/cvoc.pdf)]
* Videos of lectures at the University of Florida from the Spring of 2012. Dr. Anil V. Rao. [[web](http://www.anilvrao.com/Optimal-Control-Videos.html)]

### Books

* Bertsekas, Dimitri P., et al. Dynamic programming and optimal control. Vol. 1. No. 2. Belmont, MA: Athena scientific, 1995.
* Betts, J., Practical Methods for Optimal Control and Estimation Using Nonlinear Programming, SIAM, 2010
* Biegler, L. T., Nonlinear Programming, SIAM, 2010
* Model Predictive Control: Theory, Computation, and Design, 2nd Edition by Rawlings, Mayne, Diehl [[web](https://sites.engineering.ucsb.edu/~jbraw/mpc/)] [[pdf](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-2nd-printing.pdf)]

### Survey papers

* F. Topputo and C. Zhang, “Survey of Direct Transcription for Low-Thrust Space Trajectory Optimization with Applications,” Abstract and Applied Analysis, vol. 2014, Article ID 851720, 15 pages, 2014. [[edited](https://www.hindawi.com/journals/aaa/2014/851720/)

## Software
[mintOC](https://mintoc.de/index.php/Main_Page)
### Python

[BYU-PRISM/GEKKO: GEKKO Python for Machine Learning and Dynamic Optimization](https://github.com/BYU-PRISM/GEKKO)


[casadi/casadi: CasADi is a symbolic framework for numeric optimization implementing automatic differentiation in forward and reverse modes on sparse matrix-valued computational graphs. It supports self-contained C-code generation and interfaces state-of-the-art codes such as SUNDIALS, IPOPT etc. It can be used from C++, Python or Matlab/Octave.](https://github.com/casadi/casadi)

[ethz-adrl/control-toolbox: The Control Toolbox - An Open-Source C++ Library for Robotics, Optimal and Model Predictive Control](https://github.com/ethz-adrl/control-toolbox)

[Dynamic Optimization with pyomo.DAE — Pyomo 6.4.1 documentation](https://pyomo.readthedocs.io/en/stable/modeling_extensions/dae.html?highlight=optimal%20control)

[OpenMDAO/dymos: Open Source Optimization of Dynamic Multidisciplinary Systems](https://github.com/OpenMDAO/dymos)

[Shunichi09/PythonLinearNonlinearControl: PythonLinearNonLinearControl is a library implementing the linear and nonlinear control theories in python.](https://github.com/Shunichi09/PythonLinearNonlinearControl)

[Examples — opty 1.2.0.dev0 documentation](https://opty.readthedocs.io/en/latest/examples.html)


[wanxinjin/Pontryagin-Differentiable-Programming: A unified end-to-end learning and control framework that is able to learn a (neural) control objective function, dynamics equation, control policy, or/and optimal trajectory in a control system.](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming)


### C++
[PSOPT/psopt: PSOPT Optimal Control Software](https://github.com/PSOPT/psopt)

[loco-3d/crocoddyl: Crocoddyl is an optimal control library for robot control under contact sequence. Its solver is based on various efficient Differential Dynamic Programming (DDP)-like algorithms](https://github.com/loco-3d/crocoddyl)
### Julia
[infiniteopt/InfiniteOpt.jl: An intuitive modeling interface for infinite-dimensional optimization problems.](https://github.com/infiniteopt/InfiniteOpt.jl)

[odow/SDDP.jl: Stochastic Dual Dynamic Programming in Julia](https://github.com/odow/SDDP.jl)

[ai4energy/OptControl.jl: A tool to solve optimal control problem](https://github.com/ai4energy/OptControl.jl)

[thowell/DirectTrajectoryOptimization.jl: A Julia package for constrained trajectory optimization using direct methods.](https://github.com/thowell/DirectTrajectoryOptimization.jl)

[baggepinnen/DifferentialDynamicProgramming.jl: A package for solving Differential Dynamic Programming and trajectory optimization problems.](https://github.com/baggepinnen/DifferentialDynamicProgramming.jl)

### Matlab
DIDO

GPOCS2[Home | GPOPS-II - Next-Generation Optimal Control Software](https://www.gpops2.com/)

[Everglow0214/The_Adaptive_Dynamic_Programming_Toolbox](https://github.com/Everglow0214/The_Adaptive_Dynamic_Programming_Toolbox)

[nurkanovic/nosnoc: NOSNOC is an open source software package for NOnSmooth Numerical Optimal Control.](https://github.com/nurkanovic/nosnoc)

[OpenOCL/OpenOCL: Open Optimal Control Library for Matlab. Trajectory Optimization and non-linear Model Predictive Control (MPC) toolbox.](https://github.com/OpenOCL/OpenOCL)
### High level optimal control modeling languages and optimal control software

* Acado [[github](https://github.com/acado/acado)] [[web](http://acado.github.io/)]
* acados [[github](https://github.com/acados/acados)] [[web](http://acados.org/)]
* BOCOP [[web](https://www.bocop.org/)]
* Control toolbox, domain specific for robotics [[bitbucket](https://bitbucket.org/adrlab/ct/wiki/Home)]
* Dymos: Open-source Optimal Control for Multidisciplinary Systems [[github](https://github.com/OpenMDAO/dymos)]
* ICLOCS2 [[github](https://github.com/ImperialCollegeLondon/ICLOCS/)] [[web](http://www.ee.ic.ac.uk/ICLOCS/)]
* Modelica with JModelica [[web](https://jmodelica.org/)]
* OpenOCL [[github](https://github.com/OpenOCL/OpenOCL)] [[web](https://openocl.org/)]
* PSOPT [[github](https://github.com/PSOPT/psopt)] [[web](http://www.psopt.org/)]
* Pyomo with .DAE extension [[github](https://github.com/Pyomo/pyomo)] [[web](http://www.pyomo.org/)]
* towr, domain specific for legged robots [[github](https://github.com/ethz-adrl/towr)]
* AMPL with TACO extension (commercial)
* DIDO (commercial)
* Forces (commercial)
* GPOPS2 (commercial)
* gPROMS (commercial)
* Mujoco, domain specific for robotics/contact, simulator (commercial)
* Optimica, Dymola (commercial)
* PROPT (commercial)

### High level numerical optimization modeling languages

* CasADi [[github](https://github.com/casadi/casadi)] [[web](https://web.casadi.org/)]
* CVX, convex [[web](http://cvxr.com/cvx/)]
* Pyomo [[github](https://github.com/Pyomo/pyomo)] [[web](http://www.pyomo.org/)]
* Yalmip [[github](https://github.com/yalmip/YALMIP)] [[web](https://yalmip.github.io/)]

### Numerical optimization solver

#### Non-linear programming

* Ipopt [[github](https://github.com/coin-or/Ipopt)]
* CONOPT (commercial)
* Forces (commercial)
* KNITRO (commercial)
* Matlab fmincon (commercial)
* Snopt (commercial)
* WORHP (commercial)

#### Linear, quadratic, convex programming

* ECOS [[github](https://github.com/embotech/ecos)]
* hpipm [[github](https://github.com/giaf/hpipm)]
* Sedumi [[github](https://github.com/sqlp/sedumi)]
* qpDUNES [[github](https://github.com/jfrasch/qpDUNES)]
* qpOASES [[coin-or](https://projects.coin-or.org/qpOASES)]
* SDPT3 [[web](http://www.math.nus.edu.sg/~mattohkc/sdpt3.html)]
* CPLEX (commercial)
* Gruobi (commercial)
* MINOS (commercial)
* Mosek (commercial)

#### Integer, mixed-integer programming

* Bonmin

## Automatic differentiation

* CasADi [[github](https://github.com/casadi/casadi)] [[web](https://web.casadi.org/)]
* CppAD [[github](https://github.com/coin-or/CppAD)]
* CppADCodeGen [[github](https://github.com/joaoleal/CppADCodeGen)]
* JuliaDiff [[github](https://github.com/JuliaDiff/)] [[web](http://www.juliadiff.org/)]

## Other material

* Summer School on Numerical Optimization Software (includes a long list of solvers in the slides, see repository), Hans D. Mittelmann, Moritz Diehl [[web](https://www.syscop.de/teaching/2016/summer-school-on-numerical-optimization-software)] [[repository](https://gitlab.syscop.de/teaching/NOS_public)]
* Decision tree, benchmarks for optimization software, Hans D. Mittelmann [[web](http://plato.asu.edu/)]




## <span id="head29">3.3. Optimal Control</span>

[eleurent/phd-bibliography: References on Optimal Control, Reinforcement Learning and Motion Planning](https://github.com/eleurent/phd-bibliography)

[mintOC](https://mintoc.de/index.php/Main_Page)

Julia: Jump + InfiniteOpt

Jump is powerfull!!!

[jump-dev/JuMP.jl: Modeling language for Mathematical Optimization (linear, mixed-integer, conic, semidefinite, nonlinear)](https://github.com/jump-dev/JuMP.jl)

InfiniteOpt is powerfull!!!

[pulsipher/InfiniteOpt.jl: An intuitive modeling interface for infinite-dimensional optimization problems.](https://github.com/pulsipher/InfiniteOpt.jl)

GAMS unified software[GAMS Documentation Center](https://www.gams.com/latest/docs/index.html)

[GAMS-dev/gams.jl: A MathOptInterface Optimizer to solve JuMP models using GAMS](https://github.com/GAMS-dev/gams.jl)

Matlab: Yalmip unified[YALMIP](https://yalmip.github.io/)

Python: unified[Pyomo/pyomo: An object-oriented algebraic modeling language in Python for structured optimization problems.](https://github.com/Pyomo/pyomo)

[Solver Manuals](https://www.gams.com/latest/docs/S_MAIN.html)

Julia:

[martinbiel/StochasticPrograms.jl: Julia package for formulating and analyzing stochastic recourse models.](https://github.com/martinbiel/StochasticPrograms.jl)

[odow/SDDP.jl: Stochastic Dual Dynamic Programming in Julia](https://github.com/odow/SDDP.jl)

[PSORLab/EAGO.jl: A development environment for robust and global optimization](https://github.com/PSORLab/EAGO.jl)

[JuliaSmoothOptimizers/PDENLPModels.jl: A NLPModel API for optimization problems with PDE-constraints](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl)

[JuliaControl](https://github.com/JuliaControl)

[JuliaMPC/NLOptControl.jl: nonlinear control optimization tool](https://github.com/JuliaMPC/NLOptControl.jl)

Python:

casadi is powerful!

[python-control/python-control: The Python Control Systems Library is a Python module that implements basic operations for analysis and design of feedback control systems.](https://github.com/python-control/python-control)

[Shunichi09/PythonLinearNonlinearControl: PythonLinearNonLinearControl is a library implementing the linear and nonlinear control theories in python.](https://github.com/Shunichi09/PythonLinearNonlinearControl)

Matlab:

[OpenOCL/OpenOCL: Open Optimal Control Library for Matlab. Trajectory Optimization and non-linear Model Predictive Control (MPC) toolbox.](https://github.com/OpenOCL/OpenOCL)

[jkoendev/optimal-control-literature-software: List of literature and software for optimal control and numerical optimization.](https://github.com/jkoendev/optimal-control-literature-software)
