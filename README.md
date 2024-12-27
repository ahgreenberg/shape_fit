# shape_fit

Demonstrates an example of shape fitting with tomographic data. The fit is
performed using PyTorch's built-in automatic differentiation and optimization
routines. For this example, there are 400,000 total measurements (10 200x200
pixel images), and the model has 8,000,000 free parameters. The runtime is 
about five minutes on my MacBook Air (without GPU acceleration).
