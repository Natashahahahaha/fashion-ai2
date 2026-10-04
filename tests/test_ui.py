"""Render the Streamlit app and every page headlessly (streamlit.testing.AppTest).

Models are fakes (see conftest); the UI code, wardrobe, scoring and
generation are the real implementations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import FakeBackend, populate  # noqa: E402
from src.detection.detector import FashionDetector  # noqa: E402
from src.service import FashionAIService  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _wardrobe():
    from src.ui.pages import wardrobe_page

    wardrobe_page()


def _outfits():
    from src.ui.pages import outfits_page

    outfits_page()


def _recommendations():
    from src.ui.pages import recommendations_page

    recommendations_page()


def _diagnostics():
    from src.ui.pages import diagnostics_page

    diagnostics_page()


@pytest.fixture
def service(settings, encoder, monkeypatch):
    svc = FashionAIService(settings, encoder=encoder, detector=FashionDetector(settings, backend=FakeBackend([])))
    import src.ui.state as state

    monkeypatch.setattr(state, "get_service", lambda: svc)
    return svc


def ok(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_entry_point_starts(service):
    at = ok(AppTest.from_file(APP, default_timeout=60))
    assert any("Wardrobe" in t.value for t in at.title)
    assert any("empty" in i.value.lower() for i in at.info)


@pytest.mark.parametrize("page", [_outfits, _recommendations, _diagnostics])
def test_pages_handle_empty_wardrobe(service, page):
    at = ok(AppTest.from_function(page, default_timeout=60))
    assert not at.error


def test_wardrobe_gallery_shows_images(service):
    populate(service.manager)
    at = ok(AppTest.from_function(_wardrobe, default_timeout=60))
    assert any("10 item(s)" in s.value for s in at.subheader)
    # per-category counts are shown
    assert any("**Top** (3)" in m.value and "**Shoes** (2)" in m.value for m in at.markdown)
    # one stored image per wardrobe card
    assert len(at.image) >= 10


def test_delete_button_removes_item(service):
    populate(service.manager)
    at = ok(AppTest.from_function(_wardrobe, default_timeout=60))
    first = service.manager.list_items()[0]
    at.button(key=f"del_{first.id}").click()
    ok(at)
    assert service.manager.get_item(first.id) is None


def test_outfit_generation_flow(service):
    populate(service.manager)
    at = ok(AppTest.from_function(_outfits, default_timeout=60))
    at.selectbox[0].set_value("casual")
    at.selectbox[1].set_value("date")
    next(b for b in at.button if b.label == "Generate outfits").click()
    ok(at)
    looks = [m.value for m in at.markdown if m.value.startswith("### Look")]
    assert 1 <= len(looks) <= 5
    assert any("✓ Valid " in m.value for m in at.markdown)
    assert any(s.value.startswith(f"Showing {len(looks)} of ") for s in at.success)
    assert len(at.image) >= 2 * len(looks)  # actual garment images are displayed


def test_recommendations_and_diagnostics_with_items(service):
    populate(service.manager)
    at = ok(AppTest.from_function(_recommendations, default_timeout=60))
    assert any("%" in m.value for m in at.markdown)
    at = ok(AppTest.from_function(_diagnostics, default_timeout=60))
    assert any(m.label == "Wardrobe items" and m.value == "10" for m in at.metric)
    assert any("baseline" in i.value for i in at.info)
    assert any("Last outfit generation" in s.value for s in at.subheader)


def test_edit_panels_are_keyed_by_item(service):
    """Expander state must follow the item, otherwise after a delete another item's panel opens."""
    populate(service.manager)
    at = ok(AppTest.from_function(_wardrobe, default_timeout=60))
    keys = {e.key for e in at.expander if e.key}
    for item in service.manager.list_items():
        assert f"edit_exp_{item.id}" in keys


def _gallery_texts(at):
    return "\n".join([m.value for m in at.markdown] + [w.value for w in at.warning] + [c.value for c in at.caption])


def test_flagged_detection_review_form(service, scene):
    """The bus.jpg case: the person region detected as 'jumpsuit' is shown with its
    alternatives and flags, unticked, with the suggested category preselected."""
    from src.detection.detector import RawDetection

    path, boxes = scene
    person = (100, 40, 300, 480)
    service.detector = FashionDetector(
        service.settings,
        backend=FakeBackend(
            [
                RawDetection("jumpsuit", 0.72, person),
                RawDetection("coat", 0.20, person),
                RawDetection("jeans", 0.60, boxes["jeans"]),
            ]
        ),
    )
    dets = service.detector.detect(path)
    service.pending_test_detections = dets

    def page():
        import streamlit as st

        import src.ui.state as state
        from src.ui.pages import _detections_form

        svc = state.get_service()
        st.session_state.setdefault("detections", {"k": svc.pending_test_detections})
        _detections_form(svc, "k", svc.pending_test_detections)

    at = ok(AppTest.from_function(page, default_timeout=60))
    js = next(d for d in dets if d.label == "jumpsuit")
    jeans = next(d for d in dets if d.label == "jeans")
    assert any("1 need review" in m.value for m in at.markdown)
    assert any("Also detected as: coat (Outerwear) 0.20" in m.value for m in at.markdown)
    assert any("separate" in w.value.lower() for w in at.warning)
    assert at.checkbox(key=f"keep_k_{js.id}").value is False
    assert at.checkbox(key=f"keep_k_{jeans.id}").value is True
    assert at.selectbox(key=f"cat_k_{js.id}").value == "outerwear"  # suggested correction preselected
    # tick it with the suggested category -> stored as reviewed outerwear
    at.checkbox(key=f"keep_k_{js.id}").check()
    ok(at)
    next(b for b in at.button if b.label.startswith("Add 2")).click()
    ok(at)
    stored = {i.metadata.get("detector_label"): i for i in service.manager.list_items()}
    assert stored["jumpsuit"].category == "outerwear"
    assert stored["jumpsuit"].metadata["review_status"] == "confirmed"


def test_gallery_needs_review_filter_and_confirm(service, scene):
    from src.detection.detector import RawDetection

    path, boxes = scene
    det = FashionDetector(
        service.settings,
        backend=FakeBackend(
            [RawDetection("jumpsuit", 0.72, (100, 40, 300, 480)), RawDetection("coat", 0.2, (100, 40, 300, 480))]
        ),
    ).detect(path)
    populate(service.manager)
    flagged = service.manager.add_detections(det)[0]  # added without review
    at = ok(AppTest.from_function(_wardrobe, default_timeout=60))
    assert any("1 item(s) need review" in w.value for w in at.warning)
    at.selectbox[0].set_value("needs_review")
    ok(at)
    assert "Showing 1–1 of 1 matching item(s) (11 total)" in _gallery_texts(at)
    at.button(key=f"confirm_{flagged.id}").click()
    ok(at)
    assert service.manager.get_item(flagged.id).metadata["review_status"] == "confirmed"


def test_gallery_shows_every_item_with_display_pagination(service):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    for i in range(30):
        img = Image.new("RGB", (120, 160), tuple(int(x) for x in rng.integers(0, 256, 3)))
        service.manager.add_item(img, category=["top", "bottom", "shoes"][i % 3])
    at = ok(AppTest.from_function(_wardrobe, default_timeout=60))
    assert any("30 item(s)" in s.value for s in at.subheader)
    assert "Showing 1–24 of 30 matching item(s) (30 total)" in _gallery_texts(at)
    at.number_input[0].set_value(2)
    ok(at)
    assert "Showing 25–30 of 30 matching item(s) (30 total)" in _gallery_texts(at)
    at.selectbox[0].set_value("shoes")
    ok(at)
    assert "of 10 matching item(s)" in _gallery_texts(at)


def test_outfit_count_slider_and_funnel(service):
    populate(service.manager)
    at = ok(AppTest.from_function(_outfits, default_timeout=60))
    at.slider[0].set_value(10)
    next(b for b in at.button if b.label == "Generate outfits").click()
    ok(at)
    looks = [m.value for m in at.markdown if m.value.startswith("### Look")]
    assert 1 <= len(looks) <= 10
    if len(looks) < 10:
        assert any("strong outfit(s) found" in i.value for i in at.info)  # fewer -> explained, not padded
    body = "\n".join(m.value for m in at.markdown)
    assert "**Wardrobe items:** 10" in body and "**Valid candidates scored:**" in body
    assert service.last_outfit_diagnostics and service.last_outfit_diagnostics["returned"] == len(looks)
