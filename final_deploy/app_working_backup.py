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


def constraint_score(item_ids: list[str], occasion: str, aesthetic: str, weather: str) -> float:
    if occasion == "any" and aesthetic == "any" and weather == "any":
        return 0.5

    item_data = [item_attributes(i) for i in item_ids]
    score_parts: list[float] = []

    if occasion != "any":
        target = occasion.lower()
        vals = [
            1.0 if target in {str(x).lower() for x in d.get("occasion", [])} else 0.0
            for d in item_data
        ]
        score_parts.append(sum(vals) / len(vals))

    if aesthetic != "any":
        target = aesthetic.lower()
        vals = [
            1.0 if target in {str(x).lower() for x in d.get("style_tags", [])} else 0.0
            for d in item_data
        ]
        # Aesthetic labels are often broader than one tag, so partial support
        # is useful rather than hard-rejecting the outfit.
        score_parts.append(sum(vals) / len(vals))

    if weather != "any":
        desired = {
            "hot": {"summer"},
            "mild": {"spring", "autumn"},
            "cold": {"winter"},
        }.get(weather, set())
        vals = [
            1.0 if desired & {str(x).lower() for x in d.get("season", [])} else 0.0
            for d in item_data
        ]
        score_parts.append(sum(vals) / len(vals))

    return sum(score_parts) / len(score_parts) if score_parts else 0.5


# ============================================================
# OUTFIT SCORING
# ============================================================

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

    # ML remains the dominant signal. The other components are transparent
    # tie-breakers/ranking signals built from extracted visual attributes.
    final = 0.75 * ml + 0.10 * color + 0.10 * style + 0.05 * constraints
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
    for combo in raw:
        final, ml, color, style = rank_outfit(list(combo), occasion, aesthetic, weather)
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
            raise RuntimeError("No valid outfit candidates could be generated.")

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

if __name__ == "__main__":
    print()
    print("[app] Starting Flask server...")
    print("[app] Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
