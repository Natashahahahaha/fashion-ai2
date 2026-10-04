"""The four application pages: Wardrobe, Outfits, Item recommendations, Diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from src.attributes.style import OCCASIONS, STYLES
from src.detection.categories import SLOT_DISPLAY, SLOTS, labels_for_slot
from src.errors import DetectionError, DetectorUnavailableError, EmbeddingError, FashionAIError, InvalidImageError
from src.imaging import ALLOWED_EXTENSIONS, load_image
from src.recommendation.outfit_generator import Outfit
from src.ui.components import (
    color_swatches,
    crop_thumbnail,
    draw_detections,
    item_caption,
    item_image,
    score_bar,
    short_id,
)
from src.ui.state import service_or_stop
from src.wardrobe.manager import needs_review
from src.wardrobe.schemas import WardrobeItem

UPLOAD_TYPES = sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS)
ANY = "any"


def _slot_label(slot: str) -> str:
    return SLOT_DISPLAY.get(slot, slot)


def _flash(message: str, kind: str = "success") -> None:
    """Queue a message that survives the next st.rerun()."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def _show_flash() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        getattr(st, kind)(message)


def _added_message(items: list[WardrobeItem]) -> None:
    _flash(f"Added {len(items)} item(s): " + ", ".join(f"{it.display_name} {short_id(it)}" for it in items))
    dups = [it for it in items if it.metadata.get("possible_duplicate_of")]
    if dups:
        names = ", ".join(
            f"{it.display_name} {short_id(it)} (looks like #{it.metadata['possible_duplicate_of'][0][:4]})" for it in dups
        )
        _flash(
            f"Possible duplicates already in your wardrobe: {names}. Delete them in the gallery if they are the same garment.",
            "warning",
        )


# =============================================================== Wardrobe
def wardrobe_page() -> None:
    svc = service_or_stop()
    st.title("👕 Wardrobe")
    st.caption("Upload a photo → detect garments → choose what to keep → items are embedded with CLIP and stored.")
    _show_flash()

    _upload_section(svc)
    st.divider()
    _gallery_section(svc)


def _upload_section(svc) -> None:
    st.subheader("Add clothes")
    upload = st.file_uploader(
        "Photo of one or more clothing items (outfit photo, flat lay, product shot)",
        type=UPLOAD_TYPES,
        key="uploader",
        max_upload_size=max(1, int(round(svc.settings.max_upload_mb))),
        help=f"JPG, PNG, WEBP or BMP up to {svc.settings.max_upload_mb:g} MB (MAX_UPLOAD_MB).",
    )
    if upload is None:
        st.info("Upload an image to get started.")
        return

    data = upload.getvalue()
    if len(data) > svc.settings.max_upload_mb * 1024 * 1024:
        st.error(f"That file is {len(data) / 1e6:.1f} MB; the limit is {svc.settings.max_upload_mb:g} MB (MAX_UPLOAD_MB).")
        return
    try:
        img = load_image(data)
    except InvalidImageError as exc:
        st.error(str(exc))
        return

    key = hashlib.sha1(data).hexdigest()[:16]
    ext = Path(upload.name).suffix.lower() or ".jpg"
    stored = svc.settings.uploads_dir / f"{key}{ext}"
    if not stored.exists():
        stored.write_bytes(data)

    state = st.session_state.setdefault("detections", {})
    col_img, col_actions = st.columns([3, 2])

    with col_actions:
        if st.button("🔍 Detect clothing", type="primary", width="stretch"):
            with st.spinner("Running YOLO-World (the first run downloads model weights)…"):
                try:
                    state[key] = svc.detector.detect(stored)
                except DetectorUnavailableError as exc:
                    st.error(f"The detector is unavailable. {exc}")
                    st.info("You can still add the whole photo as a single item below.")
                except (InvalidImageError, DetectionError) as exc:
                    st.error(str(exc))
                    st.info("You can still add the whole photo as a single item below.")

    detections = state.get(key)
    with col_img:
        st.image(draw_detections(img, detections) if detections else img, width="stretch")

    if detections is not None:
        if not detections:
            st.warning(
                "No clothing was detected above the confidence threshold "
                f"({svc.detector.confidence_threshold:.2f}). Try a clearer photo, lower CONFIDENCE_THRESHOLD, "
                "or add the whole photo as one item below."
            )
        else:
            _detections_form(svc, key, detections)
        if svc.detector.last_dropped:
            st.caption(f"{svc.detector.last_dropped} low-confidence or malformed prediction(s) were ignored.")

    with st.expander("➕ Add the whole photo as a single item (no detection)", expanded=detections == []):
        st.caption("Use this for a single-garment photo, or when the detector misses an item. You choose the category.")
        c1, c2, c3 = st.columns(3)
        slot = c1.selectbox("Category", SLOTS, format_func=_slot_label, key=f"manual_slot_{key}")
        label = c2.selectbox("Type", labels_for_slot(slot) + ["other"], key=f"manual_label_{key}")
        style = c3.selectbox("Style (optional)", ["estimate with CLIP"] + list(STYLES), key=f"manual_style_{key}")
        if st.button("Add photo as item", key=f"manual_add_{key}"):
            with st.spinner("Embedding with CLIP…"):
                try:
                    item = svc.manager.add_item(
                        stored,
                        category=slot,
                        label=slot if label == "other" else label,
                        label_source="user",
                        source_image=str(stored),
                        style=None if style == "estimate with CLIP" else style,
                    )
                except (EmbeddingError, FashionAIError) as exc:
                    st.error(str(exc))
                else:
                    _added_message([item])
                    st.rerun()


def _detections_form(svc, key: str, detections) -> None:
    flagged = sum(1 for d in detections if d.needs_review)
    st.markdown(
        f"**{len(detections)} region(s) detected**"
        + (f", **{flagged} need review** (unticked; read the note, fix the category, then tick to add)." if flagged else ".")
    )
    selected, overrides = [], {}
    for i, det in enumerate(detections):
        with st.container(border=det.needs_review):
            c_img, c_info, c_pick, c_cat = st.columns([1, 3, 1, 2])
            if det.crop_path and Path(det.crop_path).is_file():
                c_img.image(crop_thumbnail(det.crop_path), width="stretch")
            info = f"**{i + 1}. {det.label}** · {_slot_label(det.category)} · score {det.confidence:.2f}"
            if det.alternatives:
                info += "  \nAlso detected as: " + ", ".join(
                    f"{a.label} ({_slot_label(a.category)}) {a.confidence:.2f}" for a in det.alternatives[:4]
                )
            c_info.markdown(info)
            for f in det.flags:
                c_info.warning(f.message)
            if det.suggested_category:
                c_info.caption(f"Suggested category: **{_slot_label(det.suggested_category)}**")
            keep = c_pick.checkbox("Add", value=not det.needs_review, key=f"keep_{key}_{det.id}")
            default = det.suggested_category if (det.needs_review and det.suggested_category) else det.category
            cat = c_cat.selectbox(
                "Category",
                SLOTS,
                index=SLOTS.index(default),
                format_func=_slot_label,
                key=f"cat_{key}_{det.id}",
            )
        if keep:
            selected.append(det)
            if cat != det.category:
                overrides[det.id] = cat

    st.caption(
        "Ticking a flagged region and adding it counts as your review: it will be used in outfits with the category you chose."
    )
    if st.button(f"Add {len(selected)} selected item(s) to wardrobe", type="primary", disabled=not selected):
        with st.spinner("Embedding with CLIP and computing attributes…"):
            try:
                added = svc.manager.add_detections(selected, overrides, reviewed=True)
            except (EmbeddingError, FashionAIError) as exc:
                st.error(str(exc))
                return
        _added_message(added)
        st.session_state["detections"].pop(key, None)
        st.rerun()


SORTS = {
    "Newest first": lambda it: it.created_at,
    "Oldest first": lambda it: it.created_at,
    "Category": lambda it: (SLOTS.index(it.category) if it.category in SLOTS else 99, it.created_at),
    "Detector score": lambda it: -(it.confidence or 0.0),
}


def _gallery_section(svc) -> None:
    items = svc.manager.list_items()
    st.subheader(f"Your wardrobe: {len(items)} item(s)")
    if not items:
        st.info("Your wardrobe is empty.")
        return
    counts = {s: sum(1 for it in items if it.category == s) for s in SLOTS}
    review = [it for it in items if needs_review(it)]
    st.markdown(" · ".join(f"**{_slot_label(s)}** ({n})" for s, n in counts.items() if n))
    if review:
        st.warning(
            f"{len(review)} item(s) need review and are not used in outfits until you confirm or correct their "
            "category (filter: 'Needs review')."
        )
    st.caption(
        "Colours are pixel statistics. Styles marked (est.) are CLIP zero-shot estimates. "
        "conf = detector score. Every item is used for recommendations; paging only affects this view."
    )

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    filt = c1.selectbox(
        "Category",
        ["all", "needs_review"] + [s for s in SLOTS if counts.get(s)],
        format_func=lambda s: {"all": "All", "needs_review": "Needs review"}.get(s, _slot_label(s)),
    )
    query = c2.text_input("Search", placeholder="label, colour, style…").strip().lower()
    sort = c3.selectbox("Sort", list(SORTS))
    per_page = c4.selectbox("Per page", [12, 24, 48, 96], index=1)

    shown = items
    if filt == "needs_review":
        shown = review
    elif filt != "all":
        shown = [it for it in items if it.category == filt]
    if query:
        shown = [
            it
            for it in shown
            if query
            in " ".join(
                str(v) for v in (it.label, it.category, it.metadata.get("color"), it.metadata.get("style"), it.id)
            ).lower()
        ]
    shown = sorted(shown, key=SORTS[sort], reverse=sort == "Newest first")
    pages = max(1, -(-len(shown) // per_page))
    page = st.number_input(f"Page (of {pages})", min_value=1, max_value=pages, value=1, step=1) if pages > 1 else 1
    start = (int(page) - 1) * per_page
    page_items = shown[start : start + per_page]
    st.caption(
        f"Showing {start + 1 if page_items else 0}–{start + len(page_items)} of {len(shown)} matching item(s) ({len(items)} total)."
    )

    for row_start in range(0, len(page_items), 4):
        cols = st.columns(4)
        for col, item in zip(cols, page_items[row_start : row_start + 4], strict=False):
            with col:
                _gallery_card(svc, item)

    with st.expander("Danger zone"):
        confirm = st.checkbox("I understand this deletes every item, image and embedding")
        if st.button("Clear wardrobe", disabled=not confirm):
            try:
                n = svc.manager.clear_wardrobe()
            except FashionAIError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop("outfit_result", None)
                _flash(f"Removed {n} item(s).")
                st.rerun()


def _gallery_card(svc, item: WardrobeItem) -> None:
    st.image(item_image(svc.manager, item), width="stretch")
    st.markdown(item_caption(item))
    st.markdown(color_swatches(item), unsafe_allow_html=True)
    if item.category not in SLOTS:
        st.warning(f"Unknown category '{item.category}': pick one below so the item can be used in outfits.")
    if needs_review(item):
        flags = item.metadata.get("review_flags") or []
        alts = item.metadata.get("alternatives") or []
        text = "**Needs review**: " + " ".join(f.get("message", "") for f in flags)
        if alts:
            text += "  \nAlso detected as: " + ", ".join(f"{a['label']} {a['confidence']:.2f}" for a in alts[:3])
        st.warning(text)
        if st.button("✓ Category is correct", key=f"confirm_{item.id}"):
            svc.manager.confirm_item(item.id)
            st.session_state.pop("outfit_result", None)
            _flash(f"Confirmed {item.display_name} {short_id(item)}.")
            st.rerun()
    with st.expander("Edit / delete", key=f"edit_exp_{item.id}"):
        suggested = item.metadata.get("suggested_category")
        default = suggested if needs_review(item) and suggested in SLOTS else item.category
        new_cat = st.selectbox(
            "Category",
            SLOTS,
            index=SLOTS.index(default) if default in SLOTS else 0,
            format_func=_slot_label,
            key=f"edit_cat_{item.id}",
        )
        style_opts = ["(keep)"] + list(STYLES)
        new_style = st.selectbox("Style", style_opts, key=f"edit_style_{item.id}")
        c1, c2 = st.columns(2)
        if c1.button("Save", key=f"save_{item.id}"):
            try:
                svc.manager.update_item(
                    item.id,
                    category=new_cat if (new_cat != item.category or needs_review(item)) else None,
                    style=None if new_style == "(keep)" else new_style,
                )
            except FashionAIError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop("outfit_result", None)
                _flash(f"Updated {item.display_name} {short_id(item)}.")
                st.rerun()
        if c2.button("🗑 Delete", key=f"del_{item.id}"):
            try:
                svc.manager.delete_item(item.id)
            except FashionAIError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop("outfit_result", None)
                _flash(f"Deleted {item.display_name} {short_id(item)}.")
                st.rerun()


# ================================================================ Outfits
COMPONENT_LABELS = {
    "plausibility": "Overall plausibility (fitted combination)",
    "learned": "Learned compatibility (uncalibrated)",
    "colour": "Colour coordination",
    "style": "Style agreement",
    "visual_similarity": "CLIP visual similarity (raw cosine)",
    "category": "Garment-type pairing",
    "context": "Style/occasion fit",
}


COMPONENT_FEATURE = {
    "learned": "learned_logit",
    "colour": "colour",
    "style": "style_sim",
    "visual_similarity": "clip_cos",
    "category": "category",
}


def _render_outfit(svc, rank: int, outfit: Outfit, ranking_features: list[str] | None = None) -> None:
    res = outfit.result
    with st.container(border=True):
        st.markdown(f"### Look {rank}")
        n = len(outfit.items)
        cols = st.columns([4 if j % 2 == 0 else 1 for j in range(2 * n - 1)])
        for j, item in enumerate(outfit.items):
            with cols[2 * j]:
                st.image(item_image(svc.manager, item), width="stretch")
                st.caption(f"{_slot_label(item.category)}: {item.display_name} `{short_id(item)}`")
            if j < n - 1:
                cols[2 * j + 1].markdown("<h2 style='text-align:center;margin-top:40%'>+</h2>", unsafe_allow_html=True)
        st.markdown("**Why this outfit**  \n" + "  \n".join(f"✓ {r}" for r in res.reasons))
        for c in res.caveats:
            st.caption(f"⚠ {c}")
        with st.expander("Score details", key=f"details_{rank}_{'_'.join(outfit.item_ids)}"):
            st.caption(
                "All values are model/heuristic ranking signals computed for this outfit (means over its item pairs), "
                "not probabilities that people would like it."
            )
            for key, label in COMPONENT_LABELS.items():
                v = res.components.get(key)
                if v is None:
                    continue
                feat = COMPONENT_FEATURE.get(key)
                if ranking_features is not None and feat and feat not in ranking_features:
                    label += " (shown for information; not used in ranking)"
                if key == "visual_similarity":
                    st.caption(f"{label}: {v:.2f}")
                else:
                    score_bar(label, v)


def outfits_page() -> None:
    svc = service_or_stop()
    st.title("✨ Outfit generator")
    items = svc.manager.list_items()
    if not items:
        st.info("Your wardrobe is empty. Add clothes on the Wardrobe page first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    style = c1.selectbox("Style", [ANY] + list(STYLES), format_func=lambda s: "Any" if s == ANY else s)
    occasion = c2.selectbox("Occasion", [ANY] + list(OCCASIONS), format_func=lambda s: "Any" if s == ANY else s)
    options: list[WardrobeItem | None] = [None] + items
    required = c3.selectbox(
        "Must include",
        options,
        format_func=lambda it: "Anything" if it is None else f"{it.display_name} ({_slot_label(it.category)}) {short_id(it)}",
    )
    k = c4.slider("Number of looks", 1, 20, 5)
    c5, c6 = st.columns(2)
    outerwear = c5.checkbox("Allow outerwear", value=True)
    accessory = c6.checkbox("Allow accessories", value=True)

    if st.button("Generate outfits", type="primary"):
        with st.spinner("Scoring every valid combination of your wardrobe…"):
            try:
                generated = svc.generator().generate_outfits(
                    k=k,
                    style=None if style == ANY else style,
                    occasion=None if occasion == ANY else occasion,
                    required_item_id=required.id if required else None,
                    include_outerwear=outerwear,
                    include_accessory=accessory,
                )
            except FashionAIError as exc:
                st.error(str(exc))
                return
        st.session_state["outfit_result"] = generated
        svc.last_outfit_diagnostics = generated.diagnostics

    result = st.session_state.get("outfit_result")
    if result is None:
        return
    d = result.diagnostics
    if result.outfits:
        st.success(
            f"Showing {len(result.outfits)} of {result.strong_candidates} strong outfit candidate(s) · "
            f"{d.get('candidates_scored', 0):,} valid combinations scored from {d.get('eligible_items', 0)} "
            f"usable item(s) ({d.get('wardrobe_items', 0)} in wardrobe)."
        )
    if d.get("excluded_items"):
        ex = d["excluded_items"]
        st.warning(
            f"{len(ex)} item(s) not used: "
            + "; ".join(f"{e['label']} #{e['id'][:4]} ({e['reason']})" for e in ex[:8])
            + (" …" if len(ex) > 8 else "")
            + ". Review them on the Wardrobe page."
        )
    if result.message:
        (st.info if result.outfits else st.error)(result.message)
    st.caption(f"Scoring: {d.get('scoring_mode', '')}.")
    for name, why in (d.get("unused_features") or {}).items():
        st.caption(f"{name.capitalize()} is shown but not used for ranking: {why.split(', so ')[0]}.")
    for i, outfit in enumerate(result.outfits, 1):
        _render_outfit(svc, i, outfit, d.get("ranking_features"))
    with st.expander("How these were chosen (pipeline counts)"):
        _funnel(d)


def _funnel(d: dict) -> None:
    if not d:
        st.caption("No outfit generation has run yet.")
        return
    rows = [
        ("Wardrobe items", d.get("wardrobe_items")),
        ("Usable for outfits", d.get("eligible_items")),
        ("Excluded", ", ".join(f"{v} {k}" for k, v in d.get("excluded", {}).items()) or "0"),
        ("Usable by category", ", ".join(f"{_slot_label(k)} {v}" for k, v in d.get("eligible_by_category", {}).items())),
        ("Structural combinations (incl. optional layers)", f"{d.get('structural_combinations', 0):,}"),
        ("Rejected by validity rules", f"{d.get('rejected_by_validity', 0):,}"),
        ("Valid candidates scored", f"{d.get('candidates_scored', 0):,}"),
        (
            "Above quality threshold" + (f" ({d['quality_threshold']:.3f})" if d.get("quality_threshold") else ""),
            d.get("above_threshold"),
        ),
        ("Returned after diversity selection", d.get("returned")),
        ("Unique items in results", d.get("unique_items_in_results")),
        ("Time", f"{d.get('elapsed_ms', 0)} ms"),
    ]
    if d.get("pruned_for_compute"):
        rows.append(("Pruned for compute", str(d["pruned_for_compute"])))
    for label, value in rows:
        st.markdown(f"- **{label}:** {value}")
    exposure = d.get("exposure") or []
    if exposure:
        st.caption("Item exposure (how often each usable item appears among strong candidates vs the returned looks):")
        st.dataframe(
            sorted(exposure, key=lambda r: (-r["selected"], -r["in_strong_candidates"])),
            hide_index=True,
            width="stretch",
        )


# ======================================================= Recommendations
def recommendations_page() -> None:
    svc = service_or_stop()
    st.title("🔗 Item recommendations")
    items = svc.manager.list_items()
    if len(items) < 2:
        st.info("Add at least two items of different categories to see recommendations.")
        return
    anchor = st.selectbox(
        "Pick an item", items, format_func=lambda it: f"{it.display_name} ({_slot_label(it.category)}) {short_id(it)}"
    )
    c1, c2, c3 = st.columns(3)
    style = c1.selectbox(
        "Style (optional)", [ANY] + list(STYLES), key="rec_style", format_func=lambda s: "Any" if s == ANY else s
    )
    occasion = c2.selectbox(
        "Occasion (optional)", [ANY] + list(OCCASIONS), key="rec_occ", format_func=lambda s: "Any" if s == ANY else s
    )
    k = c3.slider("How many", 1, max(2, len(items) - 1), min(12, max(1, len(items) - 1)))

    left, right = st.columns([1, 3])
    with left:
        st.image(item_image(svc.manager, anchor), width="stretch")
        st.markdown(item_caption(anchor))
    with right:
        with st.spinner("Scoring…"):
            try:
                recs, msg = svc.generator().recommend_for_item(
                    anchor.id, k=k, style=None if style == ANY else style, occasion=None if occasion == ANY else occasion
                )
            except FashionAIError as exc:
                st.error(str(exc))
                return
        if msg:
            st.warning(msg)
        elif recs:
            st.caption(f"Showing the top {len(recs)} usable partner item(s), ranked by plausibility.")
        for row_start in range(0, len(recs), 4):
            cols = st.columns(4)
            for col, rec in zip(cols, recs[row_start : row_start + 4], strict=False):
                with col:
                    st.image(item_image(svc.manager, rec.item), width="stretch")
                    st.markdown(f"**{rec.pair.score:.0%}** `{short_id(rec.item)}`  \n{rec.item.display_name}")
                    with st.expander("Breakdown", key=f"rec_exp_{anchor.id}_{rec.item.id}"):
                        st.caption(f"Strongest signals: {rec.pair.reasons.get('strongest_signals', '')}")
                        for name, value in rec.pair.components.items():
                            shown = "n/a (not available)" if value is None else f"{value:.2f}"
                            st.caption(f"{COMPONENT_LABELS.get(name, name)}: {shown}")


# ============================================================ Diagnostics
def diagnostics_page() -> None:
    svc = service_or_stop()
    st.title("🩺 System diagnostics")
    d = svc.diagnostics()

    di = d["device_info"]
    if di["warning"]:
        st.warning(di["warning"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Device", "CUDA" if d["device"].startswith("cuda") else "CPU", help=f"DEVICE={d['device_setting']}")
    c2.metric("Wardrobe items", d["wardrobe_items"])
    c3.metric("Usable embeddings", d["embeddings_ok"])
    parts = [f"torch {di['torch']}", f"PyTorch CUDA build: {di['torch_cuda_build'] or 'none (CPU-only wheel)'}"]
    parts.append(f"CUDA available: {di['cuda_available']}")
    if di["gpu"]:
        parts.append(f"GPU: {di['gpu']}")
    if di["vram_total_mb"]:
        parts.append(
            f"VRAM free {di['vram_free_mb']} / {di['vram_total_mb']} MB (this app: {di['vram_allocated_mb']} MB allocated)"
        )
    st.caption(" · ".join(parts))
    md = d["model_devices"]
    st.caption(f"Model devices: CLIP = {md['clip'] or 'not loaded yet'}, detector = {md['detector'] or 'not loaded yet'}")

    st.subheader("Models")
    if d["detector_loaded"]:
        det_state = "loaded"
    elif d["detector_cache_present"]:
        det_state = "ready (vocabulary cached), not loaded yet"
    elif d["detector_weights_present"]:
        det_state = "base weights on disk; the clothing vocabulary is encoded on first use"
    else:
        det_state = "weights not downloaded yet (first detection downloads them)"
    st.markdown(
        f"- **Detector**: YOLO-World `{d['detector_model']}` — {det_state}  \n"
        f"  `{d['detector_cache']}` · confidence threshold {d['confidence_threshold']}"
    )
    if d["detector_error"]:
        st.error(d["detector_error"])
    st.markdown(f"- **CLIP**: `{d['clip_model']}` — {'loaded' if d['clip_loaded'] else 'not loaded yet'}")
    c1, c2 = st.columns(2)
    if c1.button("Load detector now"):
        with st.spinner("Loading YOLO-World…"):
            try:
                svc.detector.load()
                st.success("Detector loaded.")
            except DetectorUnavailableError as exc:
                st.error(str(exc))
    if c2.button("Load CLIP now"):
        with st.spinner("Loading CLIP…"):
            try:
                st.success(f"CLIP loaded, embedding dimension {svc.encoder.dim}.")
            except EmbeddingError as exc:
                st.error(str(exc))

    st.subheader("Compatibility model")
    status = svc.learned_status()
    (st.success if status.state == "loaded" else st.error if status.state == "error" else st.info)(status.message)
    if status.model is not None:
        st.caption(
            "Metrics below were computed by the training run on the test split of the dataset it was trained on; "
            "they say nothing about other data."
        )
        st.json({"metrics": status.model.metrics, "training": status.model.training}, expanded=False)
    st.caption(f"Checkpoint path: `{d['checkpoint_path']}` · USE_LEARNED_MODEL={d['use_learned_model']}")
    model = svc.generator().model
    m = model.info.get("metrics", {})
    if model.fitted:
        st.markdown(
            f"**Outfit ranking:** logistic combination of {', '.join(model.features)}, fitted on validation pairs of "
            f"the Polyvore outfit dataset (`src/recommendation/ranker_model.json`). "
            f"Pair ROC-AUC on the held-out test split: {m.get('pair_test_roc_auc', float('nan')):.3f}. "
            f"Quality threshold {model.outfit_threshold:.3f} (chosen on validation): test real-outfit pass rate "
            f"{m.get('outfit_test_tpr', float('nan')):.0%}, random same-category swaps passing "
            f"{m.get('outfit_test_fpr', float('nan')):.0%}."
        )
        for name, why in model.info.get("dropped_features", {}).items():
            st.caption(f"{name.capitalize()} is shown but not used for ranking: {why.split(', so ')[0]}.")
    else:
        st.warning("ranker_model.json not found: outfit ranking uses unfitted equal weights and no quality threshold.")
    st.caption(f"Style/occasion preference weight (re-ranking only): {d['weights']['context']:.2f}.")

    st.subheader("Last outfit generation")
    _funnel(svc.last_outfit_diagnostics or {})

    st.subheader("Wardrobe data")
    by_cat = d["items_by_category"]
    st.caption(" · ".join(f"{_slot_label(k)}: {v}" for k, v in sorted(by_cat.items())) if by_cat else "Wardrobe is empty.")
    bad = d["embeddings_missing"] + d["embeddings_corrupt"] + d["embeddings_stale"]
    if bad:
        st.warning(
            f"{len(d['embeddings_missing'])} missing, {len(d['embeddings_corrupt'])} corrupt and "
            f"{len(d['embeddings_stale'])} stale (different CLIP model) embeddings."
        )
        if st.button("Re-embed affected items"):
            with st.spinner("Re-embedding…"):
                try:
                    n = svc.manager.reembed(bad)
                    st.success(f"Re-embedded {n} item(s).")
                except FashionAIError as exc:
                    st.error(str(exc))
    else:
        st.success("All stored embeddings are present and valid.")

    if d["missing_images"]:
        st.warning(
            f"{len(d['missing_images'])} item(s) have no image file on disk ({', '.join('#' + i[:4] for i in d['missing_images'])}). "
            "They still work in outfits if their embedding is valid, but show a placeholder; delete them from the gallery "
            "if you no longer have the photo."
        )
    if d["invalid_category"]:
        st.warning(
            f"{len(d['invalid_category'])} item(s) have an unknown category and are skipped in outfits. "
            "Fix them in the gallery (Edit / delete)."
        )
    if d["orphan_files"]:
        st.info(
            f"{d['orphan_files']} image/embedding file(s) are not referenced by any wardrobe item "
            "(for example left over after a crash)."
        )
        if st.button("Remove unreferenced files"):
            n = svc.manager.remove_orphan_files()
            st.success(f"Removed {n} file(s).")
    st.caption(f"Data directory: `{d['data_dir']}` · database: `{d['db_path']}`")
