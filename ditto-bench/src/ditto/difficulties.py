"""Canonical variant identities, kept independent of simulator imports."""
DIFFICULTIES = ("easy", "medium", "hard", "xhard")
SCORING_VERSION = "physical-relative-progress-v7"


def normalize_difficulty(value="easy"):
    value = str(value).strip().lower()
    if value not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty {value!r}; expected {DIFFICULTIES}")
    return value


def expand_difficulties(value="easy"):
    values = value.split(",") if isinstance(value, str) else value
    if any(str(v).strip().lower() == "all" for v in values):
        return DIFFICULTIES
    return tuple(dict.fromkeys(normalize_difficulty(v) for v in values))
