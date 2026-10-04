"""Full UI workflow on the real Streamlit pages with the REAL YOLO-World + CLIP models.

    RUN_MODEL_TESTS=1 pytest tests/test_real_ui_workflow.py -s

Opt-in (needs downloaded weights). Uses a temporary data directory.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.service import FashionAIService  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
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


def ok(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [s.value for s in at.success] + [w.value for w in at.warning]
    parts += [e.value for e in at.error] + [c.value for c in at.caption] + [s.value for s in at.subheader]
    parts += [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def upload_and_detect(at: AppTest, name: str) -> AppTest:
    at.file_uploader[0].upload(name, (SAMPLES / name).read_bytes(), "image/jpeg")
    ok(at)
    next(b for b in at.button if "Detect clothing" in b.label).click()
    return ok(at)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    for name in ("bus.jpg", "zidane.jpg"):
        if not (SAMPLES / name).exists():
            import urllib.request

            SAMPLES.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://ultralytics.com/images/{name}", SAMPLES / name)
    settings = dataclasses.replace(get_settings(), data_dir=tmp_path / "data")
    service = FashionAIService(settings)
    import src.ui.state as state

    monkeypatch.setattr(state, "get_service", lambda: service)
    return service


def test_full_workflow_with_real_models(svc, monkeypatch):
    at = ok(AppTest.from_function(_wardrobe, default_timeout=300))

    # corrupt upload -> readable error, no crash
    at.file_uploader[0].upload("broken.jpg", b"not an image", "image/jpeg")
    ok(at)
    assert any("not a recognised image" in e.value for e in at.error)

    # upload + detect + correct the first detection's category + add
    at = upload_and_detect(at, "bus.jpg")
    detected = [m.value for m in at.markdown if " · score " in m.value]
    assert len(detected) >= 3, detected
    # the bus.jpg 'jumpsuit' (a person in coat + jeans) is flagged, unticked, with alternatives shown
    assert any("jumpsuit" in m and "Also detected as: coat" in m for m in detected), detected
    assert any("need review" in m.value for m in at.markdown)
    cats = [s for s in at.selectbox if s.key and s.key.startswith("cat_")]
    keeps = [c for c in at.checkbox if c.key and c.key.startswith("keep_")]
    assert len(cats) == len(keeps) == len(detected)
    first_label = detected[0].split("**")[1].split(". ", 1)[1]
    assert keeps[0].value is False  # flagged -> not added unless the user decides to
    cats[0].set_value("top")
    for c in keeps:
        c.check()
    ok(at)
    next(b for b in at.button if b.label.startswith("Add ") and "selected" in b.label).click()
    ok(at)
    assert any("Added" in s.value for s in at.success), texts(at)
    items = svc.manager.list_items()
    assert len(items) == len(detected)
    corrected = [i for i in items if i.metadata.get("detector_label") == first_label and i.category == "top"]
    assert corrected and corrected[0].metadata["label_source"] == "user"
    assert corrected[0].metadata["review_status"] == "confirmed"
    assert any(f"Your wardrobe: {len(items)} item(s)" in s.value for s in at.subheader)
    assert len(at.image) >= len(items)  # every card shows its thumbnail

    # second photo
    at = upload_and_detect(at, "zidane.jpg")
    for c in [c for c in at.checkbox if c.key and c.key.startswith("keep_")]:
        c.check()
    ok(at)
    next(b for b in at.button if b.label.startswith("Add ") and "selected" in b.label).click()
    ok(at)
    total = svc.manager.count()
    assert total > len(items)

    # simulated restart: brand-new service object on the same data dir
    restarted = FashionAIService(svc.settings)
    import src.ui.state as state

    monkeypatch.setattr(state, "get_service", lambda: restarted)
    at = ok(AppTest.from_function(_wardrobe, default_timeout=300))
    assert any(f"Your wardrobe: {total} item(s)" in s.value for s in at.subheader)

    # outfits
    at = ok(AppTest.from_function(_outfits, default_timeout=300))
    next(b for b in at.button if b.label == "Generate outfits").click()
    ok(at)
    looks = [m.value for m in at.markdown if m.value.startswith("### Look")]
    body = texts(at)
    # real looks, or an explicit reason (threshold / missing garments); never padding
    assert looks or "quality threshold" in body or "Not enough items" in body or "needs shoes" in body, body
    if looks:
        assert "✓ Valid " in body and f"Showing {len(looks)} of " in body
    assert "**Wardrobe items:**" in body  # pipeline funnel
    assert "baseline" in body or "learned compatibility model" in body  # scoring mode is always stated
    for banned in ("material", "silhouette", "fabric"):
        assert banned not in body.lower()
    assert not restarted.encoder_loaded  # outfit generation never loads CLIP

    # recommendations + diagnostics
    at = ok(AppTest.from_function(_recommendations, default_timeout=300))
    assert any("%" in m.value for m in at.markdown)
    at = ok(AppTest.from_function(_diagnostics, default_timeout=300))
    assert any(m.label == "Wardrobe items" and str(m.value) == str(total) for m in at.metric)
    assert "All stored embeddings are present and valid." in texts(at)

    # delete through the gallery button
    at = ok(AppTest.from_function(_wardrobe, default_timeout=300))
    victim = restarted.manager.list_items()[0]
    at.button(key=f"del_{victim.id}").click()
    ok(at)
    assert restarted.manager.get_item(victim.id) is None
    assert any("Deleted" in s.value for s in at.success)

    # the real entry point starts with this wardrobe
    ok(AppTest.from_file(APP, default_timeout=300))
