"""The admin web UI (#70): FastAPI over the runner's control API.

The only subpackage that imports FastAPI - install it with
`pip install -e ".[web]"`. Nothing outside `web/` may import this, so the
driver stays importable without it.
"""

from df2_pi.web.app import AppContext, create_app

__all__ = ["AppContext", "create_app"]
