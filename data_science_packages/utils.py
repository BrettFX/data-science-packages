__author__ = 'Brett Allen (brettallen777@gmail.com)'

from typing import Any

class CustomConfig(dict):
    """
    Custom configuration class with custom getter function to return default value if original value is null.
    Allows dot operator access for keys in dictionary.

    Examples:
    ```python
    >>> config = CustomConfig(
        key1='value1',
        key2=23,
        key3=True
    )

    >>> print(config.key1)
    'value1'
    >>> print(config.get('not exists', 'default value'))
    'default value'
    ```
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get(self, key: Any, default: Any=None) -> Any:
        if key not in self or self[key] is None:
            return default
        return self.__getitem__(key)

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'CustomConfig' object has no attribute '{key}'")
        