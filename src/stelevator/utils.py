from collections import UserList


class _ListSameType(UserList):
    """List of objects of same type"""
    def __init__(self, dtype, data=None):
        super().__init__(data)
        if any(not isinstance(d, dtype) for d in self.data):
            raise TypeError(f"All parameters must an instance of '{dtype.__class__.__name__}'.")


# def replace_docstring(oldvalue, newvalue):
#     """Replace 'oldvalue' with 'newvalue' in the docstring of decorated function."""
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs)
#         wrapper.__doc__ = func.__doc__.replace(oldvalue, newvalue)
#         return wrapper
#     return decorator
