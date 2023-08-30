from collections import UserList


class _ListSameType(UserList):
    """List of objects of same type"""
    def __init__(self, dtype, data=None):
        super().__init__(data)
        if any(not isinstance(d, dtype) for d in self.data):
            raise TypeError(f"All parameters must an instance of '{dtype.__class__.__name__}'.")
