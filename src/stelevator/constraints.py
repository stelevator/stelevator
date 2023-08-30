from .utils import _ListSameType


class Constraint(object):
    def __call__(self, x):
        raise NotImplementedError("'Constraint' is an abstract base class.")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Real(Constraint):
    """For real numbers (does not explicitly exclude complex numbers)."""
    def __call__(self, x):
        return (x == x) & (x != float("inf")) & (x != float("-inf"))

    def __str__(self) -> str:
        return 'R'


class Interval(Constraint):
    """For the interval x ∈ [lower, upper).
    
    Args:
        lower (float): Lower bound.
        upper (float): Upper bound.
    """
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
    
    def __call__(self, x):
        return (x >= self.lower) & (x < self.upper)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.lower}, {self.upper})"

    def __str__(self) -> str:
        return f'[{self.lower}, {self.upper})'


class Boolean(Constraint):
    """For boolean values (0 or 1)."""
    def __call__(self, x):
        return (x == 0) | (x == 1)
    
    def __str__(self) -> str:
        return '{0, 1}'


class Integer(Constraint):
    """For integer values."""
    def __call__(self, x):
        return x % 1 == 0

    def __str__(self) -> str:
        return 'Z'


class IntegerInterval(Interval):
    """For the integer interval x ∈ [lower, upper).
    
    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
    """    
    def __call__(self, x):
        return super().__call__(x) & (x % 1 == 0)

    def __str__(self) -> str:
        return f'Z ∩ [{self.lower}, {self.upper})'


class ConstraintList(_ListSameType):
    """List of Constraint objects."""
    def __init__(self, constraints=None):
        super().__init__(Constraint, data=constraints)


real = Real()
interval = Interval
boolean = Boolean()
integer = Integer()
integer_interval = IntegerInterval
