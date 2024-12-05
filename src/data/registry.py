class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def inner(subclass):
            cls._registry[name] = subclass
            return subclass
        return inner

    @classmethod
    def get(cls, name):
        if name not in cls._registry:
            raise KeyError(f"Dataset '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())