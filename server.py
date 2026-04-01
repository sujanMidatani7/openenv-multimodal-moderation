from openenv.core.env_server.http_server import create_fastapi_app

from app.env import ContentModerationEnv
from app.models import Action, Observation
from app.server_routes import attach_server_routes


_ENV = ContentModerationEnv()


def _env_factory() -> ContentModerationEnv:
    return _ENV


app = create_fastapi_app(_env_factory, Action, Observation)
attach_server_routes(app, _ENV)
