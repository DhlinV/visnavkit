"""Small explicit registries; registration never imports third-party source code."""


class Registry:
    def __init__(self, name):
        self.name = name
        self._entries = {}

    def register(self, name):
        def decorator(value):
            if name in self._entries:
                raise ValueError(f"{self.name} already contains {name!r}")
            self._entries[name] = value
            return value

        return decorator

    def get(self, name):
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"Unknown {self.name} {name!r}; choices: {sorted(self._entries)}") from None

    def names(self):
        return tuple(sorted(self._entries))


PREDICTORS = Registry("predictor")
HOOKS = Registry("hook")
