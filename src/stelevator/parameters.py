import numpy as np
import astropy.units as u
from .utils import _ListSameType


class Parameter(object):
    """Base class for parameters.

    Args:
        name (str): Name of the parameter.
        symbol (str): LaTeX symbol of the parameter. Defaults to None.
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
            '_'.join([parameter.name, 'init']), 
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
            '_'.join([parameter.name, 'surf']), 
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
helium = Parameter('Y', 'Y', desc='Stellar helium mass fraction')
metals = Parameter('Z', 'Z', desc='Stellar heavy element mass fraction')
a_mlt = Parameter('a_MLT', r'\alpha_\mathrm{MLT}', desc='Mixing length parameter')
m_h = Parameter('M_H', r'[\mathrm{M}/\mathrm{H}]', 'dex', desc='Metallicity')
radius = Parameter('radius', 'R', 'Rsun', desc='Stellar radius')
luminosity = Parameter('lum', 'L', 'Lsun', desc='Stellar luminosity')
teff = Parameter('Teff', r'T_\mathrm{eff}', 'K', desc='Stellar effective temperature')
delta_nu = Parameter('delta_nu', r'\Delta\nu', 'uHz', desc='Asteroseismic large frequency separation')
p_rot = Parameter('P_rot', r'P_\mathrm{rot}', 'day', desc='Stellar rotation period')
gravity = Parameter('g', 'g', 'cm/s2', desc='Stellar surface gravity')
f_evol = Parameter('f_evol', r'f_\mathrm{evol}', desc='Fractional evolutionary phase')
