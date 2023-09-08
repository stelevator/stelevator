import os, h5py
import numpy as np
import pandas as pd
from warnings import warn
from numpy.typing import ArrayLike
from collections import namedtuple
from .parameters import (
    ParameterList, Log10, Surface, Initial,
    f_evol, mass, helium, metals, a_mlt,
    age, radius, teff, delta_nu, m_h,
)
from .constraints import ConstraintList, real, interval
from .utils import _DATADIR
from .nn import elu


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
        if domain is None:
            domain = ConstraintList([real for _ in inputs])
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

    @property
    def summary(self) -> str:
        return self._summary

    def model(self, x: ArrayLike) -> np.ndarray:
        """Returns the model output for the given input.

        This returns the raw model output, without input validation.

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        raise NotImplementedError(f"Model for '{self.__class__.__name__}' is not yet implemented.")

    def grid(self, **inputs) -> pd.DataFrame:
        """Returns a grid of model outputs for the product of the given inputs.
        
        Args:
            **inputs: Keyword arguments for the inputs to the model.
        
        Returns:
            pd.DataFrame: Table with grid inputs as the index and outputs as the columns.
        """
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

    def error(self, x: np.ndarray) -> np.ndarray:
        """Return estimate of the error at a given input. This is the truth minus the model output.

        The estimate comes from the emulators test dataset. This could output a distribution or
        parameters for a distribution.

        If the mean error is zero then its variance can just be added to that of the likelihood
        during inference.
        """
        raise NotImplementedError(f"Error for '{self.__class__.__name__}' is not yet implemented.") 

    def validate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y[~self.in_domain(x)] = np.nan
        return y

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Returns the model output for the given input.
        
        Inputs are validated against the domain of the emulator. 

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        x = np.asarray(x)
        if x.shape[-1] != len(self.inputs):
            raise ValueError(f"Input must have {len(self.inputs)} dimensions.")
        return self.validate(x, self.model(x))


class MESASolarLikeEmulator(Emulator):
    """Emulator for the MESA solar-like oscillator model from Lyttle et al. (2021)."""
    _filename = os.path.join(_DATADIR, 'lyttle21.weights.h5')
    def __init__(self):
        inputs = ParameterList([
            f_evol,
            mass,
            a_mlt,
            Initial(helium),
            Initial(metals),
        ])
        outputs = ParameterList([
            Log10(age),
            teff,
            radius,
            delta_nu,
            Surface(m_h)
        ])
        domain = ConstraintList([
            interval(0.01, 2.0),
            interval(0.8, 1.2),
            interval(1.5, 2.5),
            interval(0.22, 0.32),
            interval(0.005, 0.04),
        ])
        super().__init__(inputs, outputs, domain=domain)
        
        self.loc = namedtuple('Loc', ['inputs', 'outputs'])(
            np.array([0.865, 1.0, 1.9, 0.28, 0.017]),
            np.array([0.79, 5566.772, 1.224, 100.72, 0.081])
        )
        self.scale = namedtuple('Scale', ['inputs', 'outputs'])(
            np.array([0.651, 0.118, 0.338, 0.028, 0.011]),
            np.array([0.467, 601.172, 0.503, 42.582, 0.361])
        )
        self.weights, self.bias = self._load_weights()

    def _load_weights(self):
        """Loads the model weights and biases from file.
        
        Returns:
            tuple: Tuple of lists containing the weights and biases for each layer.
        """
        with h5py.File(self._filename, 'r') as file:
            weights = [file['dense']['dense']['kernel:0'][()]]
            bias = [file['dense']['dense']['bias:0'][()]]
            for i in range(1, 7):
                weights.append(file[f'dense_{i}'][f'dense_{i}']['kernel:0'][()])
                bias.append(file[f'dense_{i}'][f'dense_{i}']['bias:0'][()])
        return weights, bias

    def model(self, x):
        x = (x - self.loc.inputs) / self.scale.inputs
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            x = elu(np.dot(x, w) + b)
        x = np.dot(x, self.weights[-1]) + self.bias[-1]
        return self.loc.outputs + self.scale.outputs * x


class MESADeltaScutiEmulator(Emulator):
    """Emulator for the MESA δ Sct oscillator model from Scutt et al. (in review)."""


class MESARotationEmulator(Emulator):
    """Emulator for the MESA rotation model from Saunders et al. (in preparation)."""


class YRECRotationEmulator(Emulator):
    """Emulator for the YREC rotation model from Saunders et al. (in preparation)."""
