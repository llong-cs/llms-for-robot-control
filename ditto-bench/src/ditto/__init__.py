"""Lightweight catalog and lazy entry points; simulation dependencies are optional."""
__version__ = "0.7.4"


def make_env(task=0, **kwargs):
    """Create a task by zero-based ID, public slug, or registered environment ID."""
    import gymnasium as gym

    from . import envs  # noqa: F401
    from .catalog import get_task
    return gym.make(get_task(task).env_id, **kwargs)
