import numpy as np
import astropy.units as u
from .utils import _ListSameType


class Parameter(object):
    """Base class for parameters.

    Args:
        name (str): Name of the parameter.
        symbol (str): Symbol of the parameter. Defaults to None.
        unit (str or astropy.units.Unit): Unit of the parameter. Defaults to None.
        desc (str): Description of the parameter. Defaults to None.
    """
    def __init__(self, name, symbol=None, unit=None, desc=None):
        self.name = name
        self.unit = u.Unit('' if unit is None else unit)
        self.symbol = r'\mathrm{' + name + '}' if symbol is None else symbol
        self.desc = name if desc is None else desc

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', symbol='{self.symbol}', unit='{self.unit.to_string()}', desc='{self.desc}')"


class _ParameterWrapper(Parameter):
    """Abstract base class for parameter wrappers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Initial(_ParameterWrapper):
    """Base class for initial parameters."""
    def __init__(self, parameter: Parameter):
        desc = parameter.desc[0].lower() + parameter.desc[1:]
        super().__init__(
            '_'.join(['init', parameter.name]), 
            '_'.join([parameter.symbol, r'\mathrm{init}']),
            parameter.unit,
            f"Initial {desc}"
        )


class Surface(_ParameterWrapper):
    """Base class for surface parameters."""
    def __init__(self, parameter: Parameter):
        self._parameter = parameter
        desc = parameter.desc[0].lower() + parameter.desc[1:]
        super().__init__(
            '_'.join(['surf', parameter.name]), 
            '_'.join([parameter.symbol, r'\mathrm{surf}']),
            parameter.unit,
            f"Surface {desc}"
        )


class Log10(_ParameterWrapper):
    def __init__(self, parameter: Parameter):
        desc = parameter.desc[0].lower() + parameter.desc[1:]
        super().__init__(
            '_'.join(['log', parameter.name]), 
            rf'\log({parameter.symbol})',
            u.dex(parameter.unit),
            f"Base-10 logarithm of {desc}"
        )


class ParameterList(_ListSameType):
    """List of Parameter objects."""
    def __init__(self, parameters=None):
        super().__init__(Parameter, data=parameters)


mass = Parameter('mass', 'M', 'Msun', desc='Stellar mass')
age = Parameter('age', 't', 'Gyr', desc='Stellar age')
helium_mass_frac = Parameter('helium_mass_frac', 'Y', desc='Stellar helium mass fraction')
metal_mass_frac = Parameter('metal_mass_frac', 'Z', desc='Stellar heavy element mass fraction')
mixing_length_param = Parameter('mixing_length_param', r'\alpha_\mathrm{MLT}', desc='Mixing length parameter')
metal_over_hydrogen = Parameter('metal_over_hydrogen', r'[\mathrm{M}/\mathrm{H}]', 'dex', desc='Metallicity')
radius = Parameter('radius', 'R', 'Rsun', desc='Stellar radius')
luminosity = Parameter('luminosity', 'L', 'Lsun', desc='Stellar luminosity')
effective_temperature = Parameter('effective_temperature', r'T_\mathrm{eff}', 'K', desc='Stellar effective temperature')
large_freq_sep = Parameter('large_freq_sep', r'\Delta\nu', 'uHz', desc='Asteroseismic large frequency separation')
rotation_period = Parameter('rotation_period', r'P_\mathrm{rot}', 'day', desc='Stellar rotation period')
surface_gravity = Parameter('surface_gravity', 'g', 'cm/s2', desc='Stellar surface gravity')
frac_evol = Parameter('frac_evol', r'f_\mathrm{evo}', desc='Fractional evolutionary phase')
