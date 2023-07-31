import numpy as np


class Parameter(object):
    """Base class for parameters.
    
    Args:
        name (str): Name of the parameter.
        symbol (str): Symbol of the parameter. Defaults to None.
        unit (str): Unit of the parameter. Defaults to None.
        bounds (object): Bounds of the parameter. Defaults to None.
        desc (str): Description of the parameter. Defaults to None.
    """
    def __init__(self, name, symbol=None, unit=None, bounds=None, desc=None):
        self.name = name
        self.unit = unit
        self.bounds = (-np.inf, np.inf) if bounds is None else bounds
        self.symbol = symbol
        self.desc = desc


class Initial(Parameter):
    def __init__(self, parameter: Parameter):
        desc = parameter.desc[0].lower() + parameter.desc[1:]
        super().__init__(
            '_'.join([parameter.name, 'init']), 
            '_'.join([parameter.symbol, r'\mathrm{init}']),
            parameter.unit,
            parameter.bounds,
            f"Initial {desc}."
        )


class Surface(Parameter):
    def __init__(self, parameter: Parameter):
        desc = parameter.desc[0].lower() + parameter.desc[1:]
        super().__init__(
            '_'.join([parameter.name, 'surf']), 
            '_'.join([parameter.symbol, r'\mathrm{surf}']),
            parameter.unit,
            parameter.bounds,
            f"Surface {desc}."
        )


class Mass(Parameter):
    def __init__(self, unit='Msun', bounds=None):
        desc = "Stellar mass."
        super().__init__('mass', 'M', unit, bounds, desc)


class Age(Parameter):
    def __init__(self, unit='Gyr', bounds=None):
        desc = "Stellar age."
        super().__init__('age', 't', unit, bounds, desc)


class Y(Parameter):
    def __init__(self, unit=None, bounds=None):
        desc = "Initial stellar helium mass fraction."
        super().__init__('Y', 'Y', unit, bounds, desc)


class Z(Parameter):
    def __init__(self, unit=None, bounds=None):
        desc = "Initial stellar heavy element mass fraction."
        super().__init__('Z', 'Z', unit, bounds, desc)


class MixingLength(Parameter):
    def __init__(self, unit=None, bounds=None):
        super().__init__('a_MLT', r'\alpha_\mathrm{MLT}', unit, bounds)


class MH(Parameter):
    def __init__(self, unit=None, bounds=None):
        super().__init__('M_H', r'[\mathrm{M}/\mathrm{H}]', unit, bounds)


class Radius(Parameter):
    def __init__(self, unit="Rsun", bounds=None):
        desc = "Stellar radius."
        super().__init__('radius', 'R', unit, bounds, desc)


class Luminosity(Parameter):
    def __init__(self, unit="Lsun", bounds=None):
        desc = "Stellar luminosity."
        super().__init__('luminosity', 'L', unit, bounds, desc)


class Teff(Parameter):
    def __init__(self, unit="K", bounds=None):
        desc = "Stellar effective temperature."
        super().__init__('Teff', r'T_\mathrm{eff}', unit, bounds, desc)


class Logg(Parameter):
    def __init__(self, unit=None, bounds=None):
        desc = "Stellar surface gravity."
        super().__init__('log_g', r'\log(g)', unit, bounds, desc)


class DeltaNu(Parameter):
    def __init__(self, unit="muHz", bounds=None):
        desc = "Asteroseismic large frequency separation."
        super().__init__('DeltaNu', r'\Delta\nu', unit, bounds, desc)


class PRot(Parameter):
    def __init__(self, unit="days", bounds=None):
        desc = "Stellar rotation period."
        super().__init__('PRot', r'P_\mathrm{rot}', unit, bounds, desc)
