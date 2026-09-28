__all__ = ["SimulatorInterface"]


def __getattr__(name):
    if name == "SimulatorInterface":
        from .interface import SimulatorInterface

        return SimulatorInterface
    raise AttributeError(name)
