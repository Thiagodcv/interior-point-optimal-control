# interior-point-optimal-control
This repository contains my class project code for CPSC 536M: Convex Optimization. For this project, I explore the theory and application of primal-dual interior-point methods for solving problems in optimal control. I attempted two implementations: one for quadratic programming (which was succesful), and another for nonlinear programming (which is still a work in progress). I then used the former implementation to control a linear time-invariant system, and the latter to control a nonlinear system. For more information, refer to my [project report](./report.pdf).

## Features
- An implementation of a QP solver which makes use of Mehrotra's predictor-corrector algorithm.
- A WIP implementation of a solver similar to the QP one, but can handle non-affine equality constraints in order to solve nonlinear optimal control problems.
- Implementation of damped-BFGS for the NLP solver.
- Code which converts an optimal control problem into a QP or NLP problem (depending on if the system is linear or nonlinear).
- Minimal testing with high code coverage.
- Two example problems.

## Usage
```shell
git clone https://github.com/Thiagodcv/interior-point-optimal-control.git
cd interior-point-optimal-control/src/examples
julia linear_control.jl (OR) julia nonlinear_control.jl
```
Please use an environment containing the [Plots](https://docs.juliaplots.org/stable/) package in order to run the examples (if you rather not do this, you can easily comment out the few lines of code which plot the results).
