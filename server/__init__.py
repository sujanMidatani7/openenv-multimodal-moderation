__all__ = ["Action", "Observation", "State", "ContentModerationEnv"]


def __getattr__(name: str):
    if name in {"Action", "Observation", "State"}:
        from server.models import Action, Observation, State

        exports = {
            "Action": Action,
            "Observation": Observation,
            "State": State,
        }
        return exports[name]
    if name == "ContentModerationEnv":
        from server.env import ContentModerationEnv

        return ContentModerationEnv
    raise AttributeError(f"module 'server' has no attribute {name!r}")
