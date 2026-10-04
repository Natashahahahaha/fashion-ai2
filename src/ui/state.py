"""Streamlit-cached singletons. Models load once per server process."""

from __future__ import annotations

import logging

import streamlit as st

from src.config import get_settings
from src.errors import FashionAIError
from src.service import FashionAIService


@st.cache_resource(show_spinner=False)
def get_service() -> FashionAIService:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return FashionAIService(get_settings())


def service_or_stop() -> FashionAIService:
    """The service, or a readable error and ``st.stop()`` (e.g. a corrupt database)."""
    try:
        return get_service()
    except FashionAIError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # configuration errors etc.
        st.error(f"The application could not start: {type(exc).__name__}: {exc}")
        st.stop()
    raise AssertionError("unreachable")
