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


class MESADeltaScutiEmulator(Emulator):
    """Emulator for the MESA δ Sct oscillator model from Scutt et al. (in review)."""


class MESARotationEmulator(Emulator):
    """Emulator for the MESA rotation model from Saunders et al. (in preparation)."""


class YRECRotationEmulator(Emulator):
    """Emulator for the YREC rotation model from Saunders et al. (in preparation)."""
