# stelevator

The **stel**lar **ev**olution emul**ator**. Going up!

**Warning: This package is a work-in-progress, expect frequent updates and changes.**

A Python package for using stellar evolutionary model emulators. We welcome contributions from the community, including adding emulators. We also appreciate any feedback.

## Installation

To use the latest development version of the package, follow this guide. A tested early release of the package is coming soon. To contribute to and develop the package, see [Contributing](#contributing).

**Warning: The development version of this package may not be fully tested, use with caution.**

Clone the package,

```bash
git clone https://github.com/stelevator/stelevator.git
```

Install the package using `pip`,

```sh
pip install stelevator
```

## Usage

Currently, only the `MESASolarLikeEmulator` training in Lyttle et al. (2021) is available. Follow this guide to use it in your work.

### Summary

You can print a summary of the emulator,

```python
from stelevator import MESASolarLikeEmulator
em = MESASolarLikeEmulator()
print(em.summary)
```

should output,

```
MESASolarLikeEmulator
=====================

Inputs
------
f_evol ∈ [0.01, 2.0): Fractional evolutionary phase in units of .
mass ∈ [0.8, 1.2): Stellar mass in units of solMass.
a_MLT ∈ [1.5, 2.5): Mixing length parameter in units of .
Y_init ∈ [0.22, 0.32): Initial stellar helium mass fraction in units of .
Z_init ∈ [0.005, 0.04): Initial stellar heavy element mass fraction in units of .

Outputs
-------
log_age: Base-10 logarithm of stellar age in units of dex(Gyr).
Teff: Stellar effective temperature in units of K.
radius: Stellar radius in units of solRad.
delta_nu: Asteroseismic large frequency separation in units of uHz.
M_H_surf: Surface metallicity in units of dex.
```

### Evaluating the Emulator

Calling the emulator will validate your inputs against the domain of the emulator (shown in the summary). For example,

```python
x = [
    [0.5, 1.0, 2.0, 0.26, 0.02],
    [2.5, 1.0, 2.0, 0.26, 0.02],
]
y = em(x)
print(y)
```

should output,

```
[[7.36763584e-01 5.52837344e+03 9.80425974e-01 1.40095673e+02 1.33057247e-01]
 [           nan            nan            nan            nan            nan]]
```

because `f_evol` has the domain [0.01, 2.0) as described in `em.summary`.

If you would like to bypass this validation, you can call the model directly with,

```python
import numpy as np
y = em.model(np.array(x))
```

*We imported `numpy` because the model requires a `numpy.ndarray` input.*

### Generating a Grid

You can quickly generate a grid of models,

```python
grid = em.grid(
    f_evol=np.arange(0.01, 2.0, 0.1),
    mass=np.arange(0.8, 1.2, 0.1),
    a_MLT=[1.9, 2.1],
    Y_init=0.28,
    Z_init=0.02,
)
print(grid)
```

should output a `pandas.DataFrame`,

```
                                  log_age         Teff    radius    delta_nu  M_H_surf
f_evol mass a_MLT Y_init Z_init                                                       
0.01   0.8  1.9   0.28   0.02   -0.636929  4691.247007  0.724720  196.808752  0.196794
            2.1   0.28   0.02   -0.640885  4720.805985  0.718094  199.449422  0.196834
       0.9  1.9   0.28   0.02   -0.838748  5124.037398  0.800112  179.760645  0.197518
            2.1   0.28   0.02   -0.841589  5167.139841  0.789454  183.266877  0.197558
       1.0  1.9   0.28   0.02   -1.032063  5494.254178  0.890690  161.322405  0.197684
...                                   ...          ...       ...         ...       ...
1.91   0.9  2.1   0.28   0.02    1.250948  4888.575933  1.806479   53.288582  0.168462
       1.0  1.9   0.28   0.02    1.086668  4932.014276  1.947085   50.315650  0.175008
            2.1   0.28   0.02    1.085058  5022.397811  1.914786   51.498336  0.175668
       1.1  1.9   0.28   0.02    0.930783  5070.605707  2.067937   48.299067  0.180263
            2.1   0.28   0.02    0.930325  5158.620451  2.030051   49.555915  0.180833

[160 rows x 5 columns]
```

where the index represents the coordinates of the grid and the columns are its outputs.

### Coming Soon

- `jax` and `tensorflow` versions of the model for GPU acceleration
- A method which estimates the neural network error on its output

## Contributing

Fork the package and then clone it where `<your-username>` is your GitHub username,

```sh
git clone https://github.com/<your-username>/stelevator.git
```

Install the package in your favourite virtual environment. Use the 'editable' flag to ensure live changes are registered with your package manager,

```sh
pip install -e stelevator
```

Add the `stelevator` remote upstream so that you can pull changes from the main development version,

```sh
cd stelevator
git remote add upstream https://github.com/stelevator/stelevator.git
git remote -v
```

which should output something like this,

```
origin	https://github.com/<your-username>/stelevator.git (fetch)
origin	https://github.com/<your-username>/stelevator.git (push)
upstream	https://github.com/stelevator/stelevator.git (fetch)
upstream	https://github.com/stelevator/stelevator.git (push)
```
