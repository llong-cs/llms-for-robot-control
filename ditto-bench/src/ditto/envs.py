"""Import once to register all six Gymnasium / ManiSkill environments."""
# Keep registration side effects in task catalog order.
# isort: off
from .tasks import geometry_tasks, contact_tasks, unlock_task  # noqa: F401
# isort: on
