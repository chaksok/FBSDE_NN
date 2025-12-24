# FBSDNN: Solving Forward–Backward SDEs with Deep Neural Networks

This repository implements a PyTorch-based framework for solving Forward–Backward Stochastic Differential Equations (FBSDEs) using deep neural networks and Monte Carlo simulation. The method targets high-dimensional problems where classical PDE solvers become computationally infeasible.

The project is designed for research and prototyping in scientific machine learning and quantitative finance.

## Features

- Deep BSDE method implemented in PyTorch
- Automatic CPU / GPU (CUDA) support
- Flexible neural network architecture
- Mini-batch Monte Carlo training
- Support for analytical and Monte Carlo reference solutions
- Modular and extensible design

## Mathematical Background

We consider Forward–Backward Stochastic Differential Equations of the form

Forward SDE:
dX_t = σ(t, X_t, Y_t) dW_t,   X_0 = ξ

Backward SDE:
dY_t = −φ(t, X_t, Y_t, Z_t) dt + Z_t dW_t,   Y_T = g(X_T)

The solution satisfies
Y_t = u(t, X_t),   Z_t = ∇_x u(t, X_t)

A neural network is used to approximate the function u(t, x).

## Project Structure

- MyNN: neural network architecture
- FBSDataSet: Monte Carlo dataset
- FBSDNN: abstract FBSDE solver
- toy_example1: example with analytical solution
- toy_example2: example with Monte Carlo reference
- train / predict: training and inference routines

(All components are currently implemented in a single script or notebook and can be modularized if needed.)

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- NumPy
- Pandas
- Matplotlib
- Seaborn

Install dependencies with

pip install torch numpy pandas matplotlib seaborn

## Usage

### Model Definition

```python
layers = [1 + D, 64, 64, 1]

model = toy_example1(
    Xi=1.0,
    T=1.0,
    M=256,
    N=20,
    D=1,
    layers=layers,
    learning_rate=1e-3,
    r=0.05,
    sigma=0.4
)
```

### Training

```python
model.train(
    NIter=100,
    epochs=10
)
```
Training is performed using mini-batch Monte Carlo sampling.

### Prediction and Evaluation

```python
Y_pred, Y_exact = model.predict(
    Xi_star=1.0,
    K=1000
)
```
The method returns the neural network approximation and the corresponding reference solution (analytical or Monte Carlo).


## Toy Examples

Example 1: Quadratic terminal condition

g(X_T) = ||X_T||²

This example admits a closed-form solution and is used to benchmark the neural network approximation.

Example 2: Nonlinear logarithmic terminal condition

g(X_T) = log(0.5 + 0.5 ||X_T||²)

The reference solution is computed using nested Monte Carlo simulation.

Implementation Details
	•	Neural network input is the concatenation of time t and state X
	•	Activation function: GELU
	•	Optimizer: Adam
	•	Gradients computed via torch.autograd
	•	Loss function enforces both FBSDE dynamics and terminal condition consistency

## References

E, W., Han, J., and Jentzen, A. (2017). Deep learning-based numerical methods for high-dimensional parabolic PDEs and BSDEs.

Han, J., Jentzen, A., and E, W. (2018). Solving high-dimensional partial differential equations using deep learning.

Future Work
	•	Correlated and multi-dimensional Brownian motion
	•	Variance reduction techniques
	•	Adaptive time stepping
	•	Alternative backends such as TensorFlow or JAX
	•	Model checkpointing and experiment logging

## License

This project is intended for research and educational use. Please cite appropriately if used in academic or professional work.

## Author

Developed by Shuo Zhai
