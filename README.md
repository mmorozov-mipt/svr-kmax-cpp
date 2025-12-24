# svr-kmax-cpp

Support Vector Regression (SVR) implementation in C++ for approximating
the aerodynamic coefficient Kmax as a function of Mach number.

## Project Goal
The goal of this project is to train a regression model using the Support
Vector Machine (SVR) method in order to obtain a smooth, continuous and
numerically stable approximation Kmax(M) from tabular aerodynamic data.

The resulting model can be stored and later used as part of a digital
twin of an aircraft during numerical motion simulation.

## Problem Description
Given discrete tabulated values of the aerodynamic coefficient Kmax
depending on the Mach number, the task is to build a continuous function
that can be evaluated at arbitrary Mach values.

Classical polynomial approximation may be unstable for this problem,
therefore Support Vector Regression with an RBF kernel is used.

## Approach
- Regression model: epsilon-SVR
- Kernel: Radial Basis Function (RBF)
- Input feature: Mach number
- Output value: Kmax
- Library: libsvm

## Output Artifacts
- `Kmax_SVR.model` — saved trained SVR model
- `predictions.tsv` — predicted Kmax values on a dense Mach grid

## Build (macOS / Linux)
```bash
mkdir build
cd build
cmake ..
cmake --build .
./svr_kmax

## Example Use Case
The trained Kmax(M) approximation can be embedded into a flight dynamics
simulator or a digital twin to provide smooth and numerically stable
aerodynamic coefficient values during time-domain simulations.

## Disclaimer
This project is intended for educational and research purposes only.
The author assumes no responsibility for any misuse of the provided code.
