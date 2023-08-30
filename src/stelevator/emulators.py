import numpy as np
import pandas as pd
from warnings import warn
from numpy.typing import ArrayLike
from .parameters import (
    Log10, Surface, Initial, ParameterList,
    frac_evol, mass, helium_mass_frac, metal_mass_frac, mixing_length_param,
    age, radius, effective_temperature, large_freq_sep, metal_over_hydrogen,
)
from .constraints import ConstraintList, real, interval


def _elu(x, alpha=1.0):
	return x if x >= 0 else alpha*(np.exp(x) -1)


class Emulator(object):
    """Base class for emulators.

    TODO: give emulator an input domain with constraints that can be applied 
    to inputs in __call__.
    
    Args:
        inputs (ParameterList): Input parameters for emulator.
        outputs (ParameterList): Output parameters for emulator.
        domain (ConstraintList, optional): List of constraints to apply to inputs. Defaults to None (real).
    """
    def __init__(self, inputs: ParameterList, outputs: ParameterList, domain: ConstraintList=None):
        self._inputs = inputs
        self._outputs = outputs
        self._domain = domain

        # TODO: make 'in units of' optional
        name = self.__class__.__name__
        self._summary = (
            f'{name}\n'
            + '='*len(name)
            + '\n\nInputs\n------\n'
            + '\n'.join(f'{i.name} ∈ {d}: {i.desc} in units of {i.unit.to_string()}.' for i, d in zip(self.inputs, self.domain))
            + '\n\nOutputs\n-------\n'
            + '\n'.join(f'{o.name}: {o.desc} in units of {o.unit.to_string()}.' for o in self.outputs)
        )

    @property
    def inputs(self) -> ParameterList:
        return self._inputs

    @property
    def outputs(self) -> ParameterList:
        return self._outputs

    @property
    def domain(self) -> ConstraintList:
        return self._domain

    def model(self, x: ArrayLike) -> np.ndarray:
        """Returns the model output for the given input.

        This returns the raw model output, without input validation.

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        raise NotImplementedError(f"Model for '{self.__class__.__name__}' is not yet implemented.")

    def summary(self) -> None:
        """ Returns a description of the emulator."""
        print(self._summary)

    def grid(self, **inputs):
        """Returns a grid of model outputs for the product of the given inputs."""
        xi = [np.atleast_1d(inputs.pop(i.name)) for i in self.inputs]
        if len(inputs) > 0:
            warn(f"Unknown keyword arguments have been ignored: {', '.join(inputs.keys())}.")
        coords = np.stack(np.meshgrid(*xi, indexing='ij'), axis=-1)
        X = np.reshape(coords, (np.prod(coords.shape[:-1]), coords.shape[-1]))
        y = self(X)

        index = pd.MultiIndex.from_arrays(X.T, names=[i.name for i in self.inputs])
        return pd.DataFrame(y, index=index, columns=[o.name for o in self.outputs])

    def in_domain(self, x: np.ndarray) -> np.ndarray:
        return np.all([d(x[..., i]) for i, d in enumerate(self.domain)], axis=0)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Returns the model output for the given input.
        
        Inputs are validated against the bounds of the emulator.

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        x = np.asarray(x)
        if x.shape[-1] != len(self.inputs):
            raise ValueError(f"Input must have {len(self.inputs)} dimensions.")
        y = self.model(x)
        y[~self.in_domain(x)] = np.nan
        return y


class _TestEmulator(Emulator):
    def __init__(self):
        inputs = ParameterList([age, mass, metal_over_hydrogen])
        outputs = ParameterList([radius, effective_temperature])
        domain = ConstraintList([real, interval(1, 2), interval(-1, 1)])
        super().__init__(inputs, outputs, domain=domain)

    def model(self, x: np.ndarray) -> np.ndarray:
        return np.stack([np.sin(x.sum(axis=-1)), np.cos(x.sum(axis=-1))], axis=-1)


class MESASolarLikeEmulator(Emulator):
    """Emulator for the MESA solar-like oscillator model from Lyttle et al. (2021)."""
    def __init__(self):
        inputs = ParameterList([
            frac_evol,
            mass,
            mixing_length_param,
            Initial(helium_mass_frac),
            Initial(metal_mass_frac),
        ])
        outputs = ParameterList([
            Log10(age),
            effective_temperature,
            radius,
            large_freq_sep,
            Surface(metal_over_hydrogen)
        ])
        domain = ConstraintList([
            interval(0.01, 2.0),
            interval(0.8, 1.2),
            interval(1.5, 2.5),
            interval(0.22, 0.32),
            interval(0.005, 0.04),
        ])
        super().__init__(inputs, outputs, domain=domain)
        self.input_loc = np.array([0.865, 1.0, 1.9, 0.28, 0.017])
        self.input_scale = np.array([0.651, 0.118, 0.338, 0.028, 0.011])
        self.output_loc = np.array([0.79, 5566.772, 1.224, 100.72, 0.081])
        self.output_scale = np.array([0.467, 601.172, 0.503, 42.582, 0.361])
        self.weights = None  # TODO: load weights
        self.bias = None

    def model(self, x):
        x = np.divide(np.subtract(x, self.input_loc), self.inputs_scale)
        for w, b in zip(self.weights, self.bias):
            x = _elu(np.dot(x, w) + b)
        x = np.dot(x, self.weights[-1]) + self.bias[-1]
        return np.add(self.output_loc, np.multiply(self.output_scale, x))


class MESADeltaScutiEmulator(Emulator):
    """Emulator for the MESA δ Sct oscillator model from Scutt et al. (in review)."""


class MESARotationEmulator(Emulator):
    """Emulator for the MESA rotation model from Saunders et al. (in preparation)."""


class YRECRotationEmulator(Emulator):
    """Emulator for the YREC rotation model from Saunders et al. (in preparation)."""
