from .api import app, build_app

__all__ = ["app", "build_app"]

# openai_api is imported lazily by uvicorn (fakedet_vlm.serve.openai_api:app)
# to avoid pulling FastAPI when only the plain detector is used.
