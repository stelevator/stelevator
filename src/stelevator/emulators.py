import numpy as np
from numpy.typing import ArrayLike


class Emulator(object):
    """Base class for emulators.
    
    Args:
        inputs (list of Parameter): Input parameters for emulator.
        outputs (list of Parameter): Output parameters for emulator.
    """
    _inputs: list = []
    _outputs: list = []

    def __init__(self):
        self.bounds = np.stack([i.bounds for i in self.inputs], axis=1)

    @property
    def inputs(self) -> list:
        return self._inputs

    @property
    def outputs(self) -> list:
        return self._outputs

    def model(self, x: ArrayLike) -> ArrayLike:
        """Returns the model output for the given input.

        This returns the raw model output, without input validation.

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.        
        """
        raise NotImplementedError(f"Model for '{self.__class__.__name__}' is not yet implemented.")

    def describe(self) -> str:
        """ Returns a description of the emulator."""
        return (
            f"{self.__class__.__name__}\n"
            + f"Bounds: {self.bounds}\n"
        )

    def __call__(self, x: ArrayLike):
        """Returns the model output for the given input.
        
        Inputs are validated against the bounds of the emulator.

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        return np.where(
            self.bounds[0] <= x <= self.bounds[1], 
            self.model(x), 
            np.nan
        )


class MESASolarLikeEmulator(Emulator):
    """Emulator for the MESA solar-like oscillator model from Lyttle et al. (2021)."""


class MESADeltaScutiEmulator(Emulator):
    """Emulator for the MESA δ Sct oscillator model from Scutt et al. (in review)."""


class MESARotationEmulator(Emulator):
    """Emulator for the MESA rotation model from Saunders et al. (in preparation)."""


class YRECRotationEmulator(Emulator):
    """Emulator for the YREC rotation model from Saunders et al. (in preparation)."""
