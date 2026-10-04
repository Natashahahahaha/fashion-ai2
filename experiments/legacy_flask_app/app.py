from __future__ import annotations

import itertools
import json
import os
import random
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_from_directory
import torch

from model import OutfitCompatibilityNet
from ollama_client import call_ollama_chat, is_ollama_available


# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(BASE_DIR, "demo_wardrobe")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "compatibility_net.pt")
ATTRIBUTES_PATH = os.path.join(DEMO_DIR, "attributes.json")
OLLAMA_TEXT_MODEL = "llama3.2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[app] PyTorch device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[app] GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD WARDROBE
# ============================================================

def load_wardrobe():
    metadata_path = os.path.join(DEMO_DIR, "metadata.json")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    embeddings: dict[str, torch.Tensor] = {}

    for item_id in metadata:
        path = os.path.join(DEMO_DIR, "embeddings", f"{item_id}.pt")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        embeddings[item_id] = tensor.to(DEVICE)

    by_category: dict[str, list[str]] = {}
    for item_id, info in metadata.items():
        category = info.get("category", "unknown")
        by_category.setdefault(category, []).append(item_id)

    return metadata, embeddings, by_category


def load_attributes() -> dict[str, dict[str, Any]]:
    if not os.path.exists(ATTRIBUTES_PATH):
        print("[app] No attributes.json found; using metadata-only mode.")
        return {}

    try:
        with open(ATTRIBUTES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[app] loaded visual attributes for {len(data)} items")
        return data
    except Exception as exc:
        print(f"[app] attributes.json could not be loaded: {exc}")
        return {}


METADATA, EMBEDDINGS, BY_CATEGORY = load_wardrobe()
ATTRIBUTES = load_attributes()
CATEGORY_NAMES = list(BY_CATEGORY.keys())

print(
    f"[app] loaded {len(METADATA)} demo items across categories: "
    f"{CATEGORY_NAMES}"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model() -> OutfitCompatibilityNet:
    model = OutfitCompatibilityNet()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_AVAILABLE = is_ollama_available()
print(f"[app] Ollama reachable: {OLLAMA_AVAILABLE}")


# ============================================================
# ATTRIBUTE HELPERS
# ============================================================

NEUTRALS = {
    "black", "white", "gray", "grey", "beige", "cream", "ivory",
    "tan", "brown", "navy", "denim", "silver", "gold"
}

COLOR_ALIASES = {
    "grey": "gray",
    "ivory": "white",
    "cream": "beige",
    "off-white": "white",
    "charcoal": "gray",
    "maroon": "red",
    "burgundy": "red",
    "teal": "blue",
    "aqua": "blue",
}

AESTHETIC_CUES = {
    "boho": {"boho", "bohemian", "earthy", "flowy", "fringe", "floral", "paisley",
             "crochet", "woven", "embroidered", "artisan", "artisanal", "relaxed",
             "rustic", "natural", "suede", "lace", "maxi", "loose"},
    "minimal": {"minimal", "minimalist", "clean", "simple", "basic", "modern",
                "monochrome", "tailored", "sleek", "understated"},
    "classic": {"classic", "timeless", "preppy", "polished", "traditional",
                "tailored", "elegant", "business casual"},
    "feminine": {"feminine", "romantic", "girly", "soft", "elegant", "dainty",
                 "bow", "floral", "delicate", "lace", "pretty"},
    "streetwear": {"streetwear", "street", "urban", "oversized", "graphic",
                   "cargo", "sporty", "sneaker", "distressed", "athleisure"},
}

OCCASION_CUES = {
    "college": {"college", "campus", "casual", "everyday", "student", "smart-casual"},
    "work": {"work", "office", "business", "professional", "smart-casual", "formal", "business casual"},
    "casual outing": {"casual", "outing", "everyday", "weekend", "smart-casual", "college"},
    "date": {"date", "romantic", "dinner", "evening", "smart-casual", "elegant", "feminine"},
    "party": {"party", "club", "celebration", "festive", "evening", "glam", "formal"},
}

CORE_CATEGORIES = {"tops", "bottoms", "all-body", "outerwear", "shoes"}

STYLE_EXCLUSIONS = {
    "boho": {"business casual", "corporate", "formal office", "professional"},
    "streetwear": {"business casual", "corporate"},
}


def item_attributes(item_id: str) -> dict[str, Any]:
    info = METADATA[item_id]
    attrs = ATTRIBUTES.get(item_id, {})

    # Metadata stays authoritative for identity/category/title.
    return {
        "id": item_id,
        "category": info.get("category", "unknown"),
        "title": info.get("title", item_id),
        "garment_type": attrs.get("garment_type", "unknown"),
        "colors": attrs.get("colors", []),
        "pattern": attrs.get("pattern", "unknown"),
        "material": attrs.get("material", "unknown"),
        "fit": attrs.get("fit", "unknown"),
        "style_tags": attrs.get("style_tags", []),
        "formality": attrs.get("formality", "unknown"),
        "season": attrs.get("season", []),
        "occasion": attrs.get("occasion", []),
        "confidence": attrs.get("confidence", 0.0),
    }


def normalized_colors(item_id: str) -> set[str]:
    colors = set()
    for color in item_attributes(item_id)["colors"]:
        c = str(color).strip().lower()
        if c:
            colors.add(COLOR_ALIASES.get(c, c))
    return colors


def color_coherence(item_ids: list[str]) -> float:
    """Small, transparent heuristic used only as a ranking tie-breaker."""
    colors = [normalized_colors(i) for i in item_ids]
    colors = [c for c in colors if c]
    if len(colors) < 2:
        return 0.5

    score = 0.5
    pair_count = 0
    total = 0.0

    for a, b in itertools.combinations(colors, 2):
        pair_count += 1
        if a & b:
            total += 1.0
            continue
        if any(x in NEUTRALS for x in a | b):
            total += 0.72
            continue
        # Conservative fallback: distinct non-neutral colors are not
        # declared incompatible; they simply receive a neutral score.
        total += 0.5

    if pair_count:
        score = total / pair_count
    return max(0.0, min(1.0, score))


def style_coherence(item_ids: list[str]) -> float:
    tags = []
    for item_id in item_ids:
        tags.append({str(t).lower() for t in item_attributes(item_id)["style_tags"] if str(t).strip()})
    tags = [t for t in tags if t]
    if len(tags) < 2:
        return 0.5

    overlaps = []
    for a, b in itertools.combinations(tags, 2):
        overlaps.append(min(1.0, len(a & b) / 2.0))

    if not overlaps:
        return 0.5
    return max(0.0, min(1.0, 0.5 + 0.5 * (sum(overlaps) / len(overlaps))))


def _text_features(d: dict[str, Any]) -> set[str]:
    values = set()
    for key in ("garment_type", "pattern", "material", "fit", "formality"):
        value = str(d.get(key, "")).strip().lower()
        if value and value != "unknown":
            values.add(value)
    for key in ("style_tags", "occasion", "season", "colors"):
        for value in d.get(key, []) or []:
            value = str(value).strip().lower()
            if value:
                values.add(value)
    return values


def _item_request_match(d: dict[str, Any], occasion: str, aesthetic: str, weather: str) -> float:
    features = _text_features(d)
    scores = []

    if occasion != "any":
        cues = OCCASION_CUES.get(occasion, {occasion})
        direct = set(str(x).lower() for x in d.get("occasion", []) or [])
        if direct & cues:
            occ = 1.0
        elif str(d.get("formality", "unknown")).lower() in {"smart-casual", "formal"} and occasion == "work":
            occ = 0.95
        elif str(d.get("formality", "unknown")).lower() == "casual" and occasion in {"college", "casual outing"}:
            occ = 0.95
        else:
            occ = 0.20
        scores.append(occ)

    if aesthetic != "any":
        cues = AESTHETIC_CUES.get(aesthetic, {aesthetic})
        overlap = len(features & cues)
        explicit = {str(x).lower() for x in d.get("style_tags", []) or []}
        if explicit & cues:
            aest = 1.0
        elif overlap >= 2:
            aest = 0.85
        elif overlap == 1:
            aest = 0.65
        else:
            aest = 0.15

        if explicit & STYLE_EXCLUSIONS.get(aesthetic, set()):
            aest = min(aest, 0.05)
        scores.append(aest)

    if weather != "any":
        seasons = {str(x).lower() for x in d.get("season", []) or []}
        desired = {"hot": {"summer"}, "mild": {"spring", "autumn"}, "cold": {"winter"}}[weather]
        if seasons & desired:
            wx = 1.0
        elif not seasons:
            wx = 0.55
        else:
            wx = 0.20

        hard_cold_words = {"sweater", "wool", "winter", "heavy", "coat", "puffer", "fur"}
        hard_hot_words = {"tank", "shorts", "sandal", "lightweight", "linen", "summer"}
        if weather == "hot" and features & hard_cold_words:
            wx = min(wx, 0.10)
        if weather == "cold" and features & hard_hot_words:
            wx = min(wx, 0.10)
        scores.append(wx)

    return sum(scores) / len(scores) if scores else 0.5


def constraint_score(item_ids: list[str], occasion: str, aesthetic: str, weather: str) -> float:
    if occasion == "any" and aesthetic == "any" and weather == "any":
        return 0.5

    # Core clothing carries most of the request. Accessories should not be able
    # to rescue a mismatched shirt/trouser combination.
    core = [i for i in item_ids if METADATA[i].get("category") in CORE_CATEGORIES]
    accessory = [i for i in item_ids if i not in core]
    if not core:
        core = item_ids

    core_scores = [_item_request_match(item_attributes(i), occasion, aesthetic, weather) for i in core]
    core_match = sum(core_scores) / len(core_scores)

    if not accessory:
        return core_match

    acc_scores = [_item_request_match(item_attributes(i), occasion, aesthetic, weather) for i in accessory]
    accessory_match = sum(acc_scores) / len(acc_scores)
    return 0.85 * core_match + 0.15 * accessory_match


def score_outfit(item_ids: list[str]) -> float:
    pairs = list(itertools.combinations(item_ids, 2))
    if not pairs:
        return 0.0

    a = torch.stack([EMBEDDINGS[i] for i, _ in pairs]).to(DEVICE)
    b = torch.stack([EMBEDDINGS[j] for _, j in pairs]).to(DEVICE)

    with torch.inference_mode():
        scores = MODEL.predict_proba(a, b)

    return float(scores.mean().item())


def rank_outfit(item_ids: list[str], occasion: str, aesthetic: str, weather: str) -> tuple[float, float, float, float]:
    ml = score_outfit(item_ids)
    color = color_coherence(item_ids)
    style = style_coherence(item_ids)
    constraints = constraint_score(item_ids, occasion, aesthetic, weather)

    # The trained ML score remains important, but explicit user requests are
    # now a first-class ranking signal rather than a tiny tie-breaker.
    final = 0.55 * ml + 0.10 * color + 0.10 * style + 0.25 * constraints
    return final, ml, color, style


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def candidate_templates(include_accessories: bool = True):
    templates = []

    if {"tops", "bottoms", "shoes"}.issubset(BY_CATEGORY):
        templates.append(["tops", "bottoms", "shoes"])

        if include_accessories and "jewellery" in BY_CATEGORY:
            templates.append(["tops", "bottoms", "shoes", "jewellery"])
        if include_accessories and "bags" in BY_CATEGORY:
            templates.append(["tops", "bottoms", "shoes", "bags"])
        if include_accessories and {"jewellery", "bags"}.issubset(BY_CATEGORY):
            templates.append(["tops", "bottoms", "shoes", "jewellery", "bags"])

    if {"all-body", "shoes"}.issubset(BY_CATEGORY):
        templates.append(["all-body", "shoes"])
        if include_accessories and "jewellery" in BY_CATEGORY:
            templates.append(["all-body", "shoes", "jewellery"])
        if include_accessories and "bags" in BY_CATEGORY:
            templates.append(["all-body", "shoes", "bags"])

    return templates


def generate_candidate_outfits(
    num_candidates: int = 250,
    occasion: str = "any",
    aesthetic: str = "any",
    weather: str = "any",
    include_accessories: bool = True,
):
    templates = candidate_templates(include_accessories=include_accessories)
    if not templates:
        return []

    rng = random.Random()
    raw: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    # For the six-items-per-category demo, exhaustive enumeration is tiny.
    # For larger wardrobes, fall back to random sampling.
    for categories in templates:
        sizes = 1
        for category in categories:
            sizes *= len(BY_CATEGORY[category])

        if sizes <= 3000:
            pools = [BY_CATEGORY[c] for c in categories]
            for combo in itertools.product(*pools):
                canonical = tuple(combo)
                if canonical not in seen:
                    seen.add(canonical)
                    raw.append(canonical)
        else:
            attempts = min(num_candidates * 6, 5000)
            for _ in range(attempts):
                combo = tuple(rng.choice(BY_CATEGORY[c]) for c in categories)
                if combo not in seen:
                    seen.add(combo)
                    raw.append(combo)

    scored = []
    explicit = occasion != "any" or aesthetic != "any" or weather != "any"
    for combo in raw:
        req_match = constraint_score(list(combo), occasion, aesthetic, weather)

        # Never silently ignore a direct request. Reject candidates whose core
        # clothing strongly contradicts it. If nothing survives, the caller
        # will report that the current demo wardrobe has no good match.
        if explicit and req_match < 0.50:
            continue

        final, ml, color, style = rank_outfit(
            list(combo), occasion, aesthetic, weather
        )
        scored.append((combo, final, ml, color, style))

    scored.sort(key=lambda x: -x[1])
    return scored[:num_candidates]


def diversified_topk(scored, k: int = 3):
    chosen = []
    used = set()

    for row in scored:
        combo = row[0]
        signature = frozenset(
            METADATA[i].get("category", "unknown") for i in combo
        )
        if signature in used:
            continue
        chosen.append(row)
        used.add(signature)
        if len(chosen) >= k:
            break

    # If all candidates have the same category signature, still return k.
    if len(chosen) < k:
        for row in scored:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= k:
                break

    return chosen


# ============================================================
# GROUNDED EXPLANATION
# ============================================================

def template_explanation(item_ids: list[str], ml_score: float, color_score: float, style_score: float) -> str:
    names = [item_attributes(i)["title"] for i in item_ids]
    pieces = [
        f"{item_attributes(i)['category']}: {item_attributes(i)['title']}"
        for i in item_ids
    ]

    return (
        f"The compatibility model scored this outfit {ml_score:.2f}/1.0. "
        f"The visual attribute layer found color coherence of {color_score:.2f} "
        f"and style coherence of {style_score:.2f}. "
        f"Items used: {', '.join(names)}."
    )


def explain_outfit(item_ids: list[str], ml_score: float, color_score: float, style_score: float) -> str:
    if not OLLAMA_AVAILABLE:
        return template_explanation(item_ids, ml_score, color_score, style_score)

    grounded_items = [item_attributes(i) for i in item_ids]
    facts = json.dumps(grounded_items, indent=2, ensure_ascii=False)

    prompt = f"""
You are the explanation layer for a fashion recommendation system.

A trained compatibility model scored the outfit {ml_score:.3f}/1.0.
Additional deterministic signals:
- color coherence: {color_score:.3f}
- style coherence: {style_score:.3f}

The following JSON contains the ONLY visual facts you may use about the items:
{facts}

Write 2-4 concise sentences explaining why the outfit works.
Rules:
- Treat the JSON as ground truth.
- Never invent a color, garment type, pattern, silhouette, material, occasion, or style tag.
- Never confuse an item's category with another item's category.
- If a useful attribute is unknown, simply omit it.
- Mention specific items by their catalog title when useful.
- Do not repeat the numeric scores unless it helps explain the recommendation.
- Do not say you looked at the image; the image analysis has already been performed.
"""

    try:
        result = call_ollama_chat(OLLAMA_TEXT_MODEL, prompt)
        return result.strip()
    except RuntimeError as exc:
        print(f"[app] Ollama call failed; using grounded template: {exc}")
        return template_explanation(item_ids, ml_score, color_score, style_score)


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_outfit(row, occasion: str, aesthetic: str, weather: str):
    combo, final, ml, color, style = row
    constraints = constraint_score(list(combo), occasion, aesthetic, weather)

    items = []
    for item_id in combo:
        data = item_attributes(item_id)
        data["image"] = f"/images/{item_id}.jpg"
        items.append(data)

    return {
        "rank_score": round(final, 3),
        "compatibility_score": round(ml, 3),
        "color_coherence": round(color, 3),
        "style_coherence": round(style, 3),
        "constraint_match": round(constraints, 3),
        "items": items,
        "explanation": explain_outfit(list(combo), ml, color, style),
    }


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Personal Style Intelligence</title>
<style>
:root { --ink:#1d1d1f; --muted:#6d6d72; --card:#fff; --bg:#f7f5f2; --line:#e8e4df; }
* { box-sizing:border-box; }
body { margin:0; font-family:Inter,Arial,sans-serif; background:var(--bg); color:var(--ink); }
.wrap { max-width:1180px; margin:0 auto; padding:38px 22px 70px; }
.hero { padding:20px 0 26px; }
h1 { font-size:42px; margin:0 0 8px; letter-spacing:-1.4px; }
.subtitle { color:var(--muted); font-size:17px; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:20px; margin:20px 0 26px; box-shadow:0 8px 28px rgba(0,0,0,.04); }
.controls { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; align-items:end; }
label { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:6px; }
select { width:100%; padding:12px 14px; border:1px solid var(--line); border-radius:10px; background:white; font-size:15px; }
button { margin-top:18px; background:#1d1d1f; color:white; border:0; padding:13px 20px; border-radius:11px; font-size:15px; cursor:pointer; }
button:disabled { opacity:.55; cursor:wait; }
.status { color:var(--muted); min-height:24px; margin:14px 0; }
.outfit { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:16px; }
.item { background:white; border:1px solid var(--line); border-radius:15px; overflow:hidden; }
.item img { display:block; width:100%; height:235px; object-fit:contain; background:#f1efec; }
.item-body { padding:14px; }
.category { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.title { margin-top:5px; font-weight:700; font-size:14px; }
.tags { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
.tag { background:#f1efec; border-radius:999px; padding:4px 7px; font-size:11px; color:#555; }
.results { margin-top:28px; }
.look { background:white; border:1px solid var(--line); border-radius:18px; padding:18px; margin-bottom:24px; }
.look-head { display:flex; justify-content:space-between; gap:15px; align-items:center; flex-wrap:wrap; margin-bottom:16px; }
.look-title { font-size:20px; font-weight:800; }
.score { font-size:14px; color:var(--muted); }
.score strong { color:var(--ink); font-size:18px; }
.breakdown { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.metric { padding:7px 10px; background:#f5f2ef; border-radius:9px; font-size:12px; }
.explanation { margin-top:16px; background:#faf9f7; border-radius:12px; padding:15px; line-height:1.55; }
.note { color:var(--muted); font-size:12px; margin-top:10px; }
@media(max-width:800px){ .controls{grid-template-columns:1fr 1fr;} h1{font-size:34px;} }
@media(max-width:520px){ .controls{grid-template-columns:1fr;} }
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Personal Style Intelligence</h1>
    <div class="subtitle">Visual wardrobe understanding + compatibility ML + grounded AI styling</div>
  </div>

  <div class="panel">
    <div class="controls">
      <div>
        <label>Occasion</label>
        <select id="occasion">
          <option value="any">Any</option>
          <option value="college">College</option>
          <option value="work">Work</option>
          <option value="casual outing">Casual outing</option>
          <option value="date">Date</option>
          <option value="party">Party</option>
        </select>
      </div>
      <div>
        <label>Aesthetic</label>
        <select id="aesthetic">
          <option value="any">Any</option>
          <option value="boho">Boho</option>
          <option value="minimal">Minimal</option>
          <option value="classic">Classic</option>
          <option value="feminine">Feminine</option>
          <option value="streetwear">Streetwear</option>
        </select>
      </div>
      <div>
        <label>Weather</label>
        <select id="weather">
          <option value="any">Any</option>
          <option value="hot">Hot</option>
          <option value="mild">Mild</option>
          <option value="cold">Cold</option>
        </select>
      </div>
      <div>
        <label>Accessories</label>
        <select id="accessories">
          <option value="yes">Use accessories</option>
          <option value="no">Clothing + shoes only</option>
        </select>
      </div>
    </div>
    <button id="generate" onclick="generateOutfits()">Generate looks</button>
    <div id="status" class="status"></div>
    <div class="note">Compatibility is produced by the trained PyTorch model; visual attributes come from the local vision model.</div>
  </div>

  <div id="results" class="results"></div>
</div>

<script>
function esc(value) {
  return String(value ?? '').replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
}

async function generateOutfits() {
  const button = document.getElementById('generate');
  const status = document.getElementById('status');
  const results = document.getElementById('results');

  button.disabled = true;
  status.textContent = 'Retrieving, scoring, and explaining looks...';
  results.innerHTML = '';

  try {
    const response = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        occasion: document.getElementById('occasion').value,
        aesthetic: document.getElementById('aesthetic').value,
        weather: document.getElementById('weather').value,
        include_accessories: document.getElementById('accessories').value === 'yes'
      })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Server error');

    data.looks.forEach((look, index) => {
      let html = '<div class="look">';
      html += '<div class="look-head">';
      html += '<div class="look-title">Look ' + (index + 1) + '</div>';
      html += '<div class="score">Rank score <strong>' + esc(look.rank_score) + '</strong> · compatibility ' + esc(look.compatibility_score) + '/1.0</div>';
      html += '</div>';

      html += '<div class="breakdown">';
      html += '<div class="metric">Color coherence ' + esc(look.color_coherence) + '</div>';
      html += '<div class="metric">Style coherence ' + esc(look.style_coherence) + '</div>';
      html += '<div class="metric">Constraint match ' + esc(look.constraint_match) + '</div>';
      html += '</div>';

      html += '<div class="outfit">';
      look.items.forEach(item => {
        html += '<div class="item">';
        html += '<img src="' + esc(item.image) + '" alt="' + esc(item.title) + '">';
        html += '<div class="item-body">';
        html += '<div class="category">' + esc(item.category) + '</div>';
        html += '<div class="title">' + esc(item.title) + '</div>';
        const tags = [...(item.colors || []).slice(0,3), ...(item.style_tags || []).slice(0,2)];
        if (tags.length) {
          html += '<div class="tags">';
          tags.forEach(t => html += '<span class="tag">' + esc(t) + '</span>');
          html += '</div>';
        }
        html += '</div></div>';
      });
      html += '</div>';
      html += '<div class="explanation"><strong>Why this look?</strong><br>' + esc(look.explanation) + '</div>';
      html += '</div>';
      results.innerHTML += html;
    });

    status.textContent = data.looks.length + ' distinct looks generated.';
  } catch (error) {
    console.error(error);
    status.textContent = 'Error: ' + error.message;
  } finally {
    button.disabled = false;
  }
}
</script>
</body>
</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "ollama": OLLAMA_AVAILABLE,
        "items": len(METADATA),
        "attributes": len(ATTRIBUTES),
    })


@app.route("/generate", methods=["GET", "POST"])
def generate():
    try:
        payload = request.get_json(silent=True) or {}
        occasion = str(payload.get("occasion", "any")).strip().lower()
        aesthetic = str(payload.get("aesthetic", "any")).strip().lower()
        weather = str(payload.get("weather", "any")).strip().lower()
        include_accessories = bool(payload.get("include_accessories", True))

        scored = generate_candidate_outfits(
            num_candidates=100,
            occasion=occasion,
            aesthetic=aesthetic,
            weather=weather,
            include_accessories=include_accessories,
        )

        if not scored:
            raise RuntimeError(
                "No outfit in the current demo wardrobe satisfies that combination of "
                f"occasion={occasion}, aesthetic={aesthetic}, weather={weather}. "
                "Try relaxing one filter rather than silently ignoring your request."
            )

        looks = diversified_topk(scored, k=3)
        serialized = [
            serialize_outfit(row, occasion, aesthetic, weather)
            for row in looks
        ]

        return jsonify({
            "device": str(DEVICE),
            "looks": serialized,
        })

    except Exception as exc:
        print("[app] generation error:", repr(exc))
        return jsonify({"error": str(exc)}), 500


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(
        os.path.join(DEMO_DIR, "images"),
        filename
    )


# ============================================================
# START SERVER
# ============================================================



from pathlib import Path
import uuid

from flask import request

from embedding_backend import encode_image
from vision_pipeline import VisionPipeline, UPLOAD_DIR

BASE_DIR = Path(__file__).resolve().parent
WARDROBE_DIR = BASE_DIR / "demo_wardrobe"
IMAGE_DIR = WARDROBE_DIR / "images"
EMBED_DIR = WARDROBE_DIR / "embeddings"
METADATA_PATH = WARDROBE_DIR / "metadata.json"
ATTRIBUTES_PATH = WARDROBE_DIR / "attributes.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {"jpg", "jpeg", "png", "webp"}
VISION = VisionPipeline()


def reload_core_wardrobe() -> None:
    global METADATA, EMBEDDINGS, BY_CATEGORY, ATTRIBUTES, CATEGORY_NAMES
    metadata, embeddings, by_category = load_wardrobe()
    METADATA = metadata
    EMBEDDINGS = embeddings
    BY_CATEGORY = by_category
    ATTRIBUTES = load_attributes()
    CATEGORY_NAMES = list(by_category.keys())
    print(f"[app] wardrobe reloaded: {len(metadata)} items")


def _safe_ext(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in ALLOWED:
        raise ValueError("Only JPG, JPEG, PNG, and WEBP images are supported.")
    return suffix


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _category_from_attrs(attrs: dict) -> str:
    category = str(attrs.get("category", "unknown")).lower().strip()
    allowed = {"tops", "bottoms", "all-body", "outerwear", "shoes", "bags", "jewellery", "accessories", "hats", "scarves"}
    return category if category in allowed else "accessories"


def process_upload(file_storage):
    ext = _safe_ext(file_storage.filename or "")
    item_id = f"user_{uuid.uuid4().hex[:12]}"
    original_path = UPLOAD_DIR / f"{item_id}.{ext}"
    file_storage.save(original_path)

    print(f"[upload] saved {original_path.name}")
    result = VISION.process(original_path)
    attrs = result["attributes"]
    category = _category_from_attrs(attrs)

    # The compatibility model expects the same 512-d image embedding space as
    # the existing wardrobe. CLIP ViT-B/32 provides a 512-d image projection.
    embedding = encode_image(original_path)
    if embedding.numel() != 512:
        raise RuntimeError(f"Embedding dimension is {embedding.numel()}, expected 512.")

    image_path = IMAGE_DIR / f"{item_id}.jpg"
    from PIL import Image
    image = Image.open(original_path).convert("RGB")
    image.save(image_path, quality=95)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    attributes = {}
    if ATTRIBUTES_PATH.exists():
        attributes = json.loads(ATTRIBUTES_PATH.read_text(encoding="utf-8"))

    title = attrs.get("garment_type", "uploaded item")
    metadata[item_id] = {
        "category": category,
        "title": f"My {title}",
        "source": "user_upload",
        "original_filename": file_storage.filename,
    }
    attributes[item_id] = {
        **attrs,
        "vision": result["vision"],
    }

    _write_json(METADATA_PATH, metadata)
    _write_json(ATTRIBUTES_PATH, attributes)
    torch_path = EMBED_DIR / f"{item_id}.pt"
    import torch
    torch.save(embedding, torch_path)

    reload_core_wardrobe()

    return {
        "id": item_id,
        "image": f"/images/{item_id}.jpg",
        "metadata": metadata[item_id],
        "attributes": attributes[item_id],
        "vision": result["vision"],
    }


UPLOAD_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personal Style Intelligence</title>
<style>
body{font-family:Arial,sans-serif;background:#f7f5f2;color:#202020;margin:0}.wrap{max-width:1180px;margin:auto;padding:32px 20px 70px}
h1{margin:0 0 6px}.muted{color:#707070}.panel{background:white;border:1px solid #e4e0dc;border-radius:18px;padding:20px;margin:20px 0}
.drop{border:2px dashed #bbb;border-radius:15px;padding:35px;text-align:center;background:#faf9f7}.drop input{margin-top:15px}.btn{border:0;border-radius:10px;padding:12px 18px;background:#202020;color:#fff;cursor:pointer;margin-top:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:15px}.card{background:#fff;border:1px solid #e5e1dd;border-radius:14px;overflow:hidden}.card img{width:100%;height:220px;object-fit:contain;background:#f1efec}.body{padding:12px}.tag{display:inline-block;background:#f1efec;border-radius:999px;padding:4px 7px;margin:4px 3px 0 0;font-size:11px}
.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.controls select{padding:11px;border:1px solid #ddd;border-radius:9px}.status{margin-top:12px;color:#666}
@media(max-width:750px){.controls{grid-template-columns:1fr 1fr}}
</style></head>
<body><div class="wrap">
<h1>Personal Style Intelligence</h1><div class="muted">Your wardrobe → vision analysis → compatibility → outfit generation</div>
<div class="panel"><h2>1. Add wardrobe items</h2><div class="drop">Upload one or multiple clothing photos.<br><input id="files" type="file" accept="image/*" multiple><br><button class="btn" onclick="upload()">Analyze wardrobe</button><div id="uploadStatus" class="status"></div></div></div>
<div class="panel"><h2>2. Generate looks</h2><div class="controls"><div><label>Occasion<br><select id="occasion"><option value="any">Any</option><option value="college">College</option><option value="work">Work</option><option value="casual outing">Casual outing</option><option value="date">Date</option><option value="party">Party</option></select></label></div><div><label>Aesthetic<br><select id="aesthetic"><option value="any">Any</option><option value="boho">Boho</option><option value="minimal">Minimal</option><option value="classic">Classic</option><option value="feminine">Feminine</option><option value="streetwear">Streetwear</option></select></label></div><div><label>Weather<br><select id="weather"><option value="any">Any</option><option value="hot">Hot</option><option value="mild">Mild</option><option value="cold">Cold</option></select></label></div><div><label>Accessories<br><select id="accessories"><option value="yes">Use accessories</option><option value="no">Clothing + shoes only</option></select></label></div></div><button class="btn" onclick="generate()">Generate looks</button><div id="genStatus" class="status"></div><div id="results"></div></div>
<div class="panel"><h2>Wardrobe</h2><div id="wardrobe" class="grid"></div></div>
</div>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function upload(){const fs=document.getElementById('files').files;if(!fs.length)return;const st=document.getElementById('uploadStatus');st.textContent='Running detector + segmentation + vision analysis + embedding...';for(const f of fs){const fd=new FormData();fd.append('image',f);try{const r=await fetch('/upload',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw Error(d.error||'upload failed');st.textContent='Analyzed '+d.metadata.title+' ('+d.metadata.category+')';}catch(e){st.textContent='Error: '+e.message;break;}}await loadWardrobe();}
async function loadWardrobe(){const r=await fetch('/wardrobe');const d=await r.json();document.getElementById('wardrobe').innerHTML=d.items.map(x=>'<div class="card"><img src="'+esc(x.image)+'"><div class="body"><b>'+esc(x.title)+'</b><div class="muted">'+esc(x.category)+'</div><div>'+[...(x.colors||[]),...(x.style_tags||[])].slice(0,5).map(t=>'<span class="tag">'+esc(t)+'</span>').join('')+'</div></div></div>').join('');}
async function generate(){const st=document.getElementById('genStatus');st.textContent='Generating...';const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({occasion:occasion.value,aesthetic:aesthetic.value,weather:weather.value,include_accessories:accessories.value==='yes'})});const d=await r.json();if(!r.ok){st.textContent='Error: '+d.error;return;}st.textContent=d.looks.length+' looks generated.';document.getElementById('results').innerHTML=d.looks.map((l,i)=>'<div class="panel"><h3>Look '+(i+1)+' · '+esc(l.rank_score)+'</h3><div class="grid">'+l.items.map(x=>'<div class="card"><img src="'+esc(x.image)+'"><div class="body"><b>'+esc(x.title)+'</b><div class="muted">'+esc(x.category)+'</div></div></div>').join('')+'</div><p>'+esc(l.explanation)+'</p></div>').join('');}
loadWardrobe();
</script></body></html>
"""

# app_v3's index route reads HTML dynamically, so replacing it here would
# also replace the UI without duplicating the Flask application object.
HTML = UPLOAD_HTML


@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify({"error": "No image supplied."}), 400
        return jsonify(process_upload(file))
    except Exception as exc:
        print("[upload] ERROR:", repr(exc))
        return jsonify({"error": str(exc)}), 500


@app.route("/wardrobe")
def wardrobe():
    items = []
    for item_id in METADATA:
        d = item_attributes(item_id)
        d["image"] = f"/images/{item_id}.jpg"
        items.append(d)
    return jsonify({"count": len(items), "items": items})


@app.route("/upload-health")
def upload_health():
    return jsonify({
        "ok": True,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "yolo_checkpoint": str(VISION.yolo_path) if VISION.yolo_path else None,
        "sam_checkpoint": str(VISION.sam_path) if VISION.sam_path else None,
        "yolo_loaded": VISION.yolo is not None,
        "sam_loaded": VISION.sam_predictor is not None,
        "ollama": OLLAMA_AVAILABLE,
        "items": len(METADATA),
    })


# /images already exists in app_v3 and serves the generated wardrobe images.

if __name__ == "__main__":
    print("[app_v4] upload-enabled Flask server")
    print(f"[app_v4] PyTorch device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[app_v4] GPU: {torch.cuda.get_device_name(0)}")
    app.run(host="127.0.0.1", port=5000, debug=False)
