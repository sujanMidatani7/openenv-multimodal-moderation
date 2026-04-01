from openenv.core.env_server.http_server import create_fastapi_app

from server.env import ContentModerationEnv, Action, Observation
from server.server_routes import attach_server_routes


_ENV = ContentModerationEnv()


def _env_factory() -> ContentModerationEnv:
    return _ENV


app = create_fastapi_app(_env_factory, Action, Observation)
attach_server_routes(app, _ENV)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
