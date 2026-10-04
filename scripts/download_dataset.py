"""Download, verify and prepare a public outfit-compatibility dataset (explicit setup step).

    python scripts/download_dataset.py                 # default: maryland-polyvore
    python scripts/download_dataset.py --list
    python scripts/download_dataset.py --yes           # skip the size confirmation

Default dataset: Maryland Polyvore (Han et al., "Learning Fashion Compatibility
with Bidirectional LSTMs", ACM MM 2017):
  * outfits + official train/valid/test split: github.com/xthan/polyvore-dataset (Apache-2.0)
  * item images: huggingface.co/datasets/Marqo/polyvore (Apache-2.0, ungated mirror of
    the same items, ids "<set_id>_<index>")
Both sources are pinned to exact revisions and verified (sizes, SHA-256).

The product images originate from Polyvore.com; their copyright may belong to
third parties. Use them for research and do not redistribute them.

Steps: download (resumable) -> verify -> safe extraction -> convert to the
Polyvore-outfits layout used by this project -> write manifest.json -> run the
dataset audit (artifacts/dataset_audit.json + .md). Nothing is downloaded by
the Streamlit app itself.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap

PREPROCESSING_VERSION = "maryland-polyvore/v1"
CONFIRM_ABOVE_BYTES = 5 * 2**30  # ask before downloading more than 5 GiB

MARQO_REV = "8c782ee447faf2d2a0402ac883cf07d3b3f43e1c"
GITHUB_REV = "ba8aa71b6589e8920d172a8b5df84746be724d00"

DATASETS: dict[str, dict[str, Any]] = {
    "maryland-polyvore": {
        "name": "Maryland Polyvore (Han et al. 2017) + Marqo/polyvore images",
        "citation": "Han, Wu, Jiang, Davis. Learning Fashion Compatibility with Bidirectional LSTMs. ACM Multimedia 2017.",
        "source": "https://github.com/xthan/polyvore-dataset ; https://huggingface.co/datasets/Marqo/polyvore",
        "license": "Apache-2.0 (both repositories). Product images originate from Polyvore.com and may be third-party "
        "copyright: research use, no redistribution.",
        "version": f"github@{GITHUB_REV[:12]} + hf@{MARQO_REV[:12]}",
        "files": [
            {
                "url": f"https://raw.githubusercontent.com/xthan/polyvore-dataset/{GITHUB_REV}/polyvore.tar.gz",
                "path": "polyvore.tar.gz",
                "size": 8412072,
                "sha256": None,  # not published upstream; computed and recorded in the manifest
            },
            *[
                {
                    "url": f"https://huggingface.co/datasets/Marqo/polyvore/resolve/{MARQO_REV}/data/data-0000{i}-of-00006.parquet",
                    "path": f"images_parquet/data-0000{i}-of-00006.parquet",
                    "size": size,
                    "sha256": sha,
                }
                for i, (size, sha) in enumerate(
                    [
                        (428492230, "c5f9570cecd73ad8050609627f121bc69801905931d227775ae10370c43b6f3b"),
                        (421082894, "4d9ecaa627db670178f3d8d372cc1b4eb64442c301c935eda0da85aa98809df6"),
                        (415568757, "fce678b1acbd90fe485695817b45f4f295420c663a8fdab992aa1e3f9a1f011e"),
                        (415941848, "946ba30f80a1fe755a8e43697766cf2cc3d7f4267a9d0be5345d3f2c5a2495b8"),
                        (421843702, "55977c926d791323a225e81ab5cefb0d43d24c2bc80861883adcd341fbc82a8b"),
                        (409375762, "f612817cb6a26d5b438d354de805a77676652666e55989511735d07bf1b2b9c8"),
                    ]
                )
            ],
        ],
    }
}


class DownloadError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None, sha256: str | None, retries: int = 3) -> str:
    """Resumable download to ``dest`` (via ``dest.part``). Returns the file's SHA-256."""
    if dest.is_file() and (size is None or dest.stat().st_size == size):
        digest = sha256_of(dest)
        if sha256 is None or digest == sha256:
            print(f"  ok (already downloaded) {dest.name}")
            return digest
        print(f"  checksum mismatch for existing {dest.name}; downloading again")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "fashion-ai-dataset-downloader/1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored Range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                total = (have + int(resp.headers.get("Content-Length", 0))) or size
                mode = "ab" if have else "wb"
                done, t0, last = have, time.time(), 0.0
                with part.open(mode) as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 2 or done == total:
                            last = time.time()
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 2**20
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            print(f"\r  {dest.name}: {done / 2**20:8.1f} MB {pct} {rate:6.1f} MB/s", end="", flush=True)
            print()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"\n  attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise DownloadError(
                    f"Could not download {url}: {exc}. The partial file {part} is kept; re-run the command to resume."
                ) from exc
            time.sleep(3 * attempt)
    if size is not None and part.stat().st_size != size:
        raise DownloadError(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}. Delete {part} and re-run.")
    digest = sha256_of(part)
    if sha256 is not None and digest != sha256:
        part.unlink()
        raise DownloadError(f"{dest.name}: SHA-256 mismatch (expected {sha256}, got {digest}). The file was removed; re-run.")
    part.replace(dest)
    return digest


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Extract a tar.gz refusing absolute paths, '..', links and special files."""
    dest = dest.resolve()
    names = []
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest)) or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise DownloadError(f"Refusing unsafe archive member {m.name!r} in {archive.name}")
        for m in members:
            tar.extract(m, dest)  # validated above
            names.append(m.name)
    return names


def prepare_maryland(raw: Path, out: Path) -> dict[str, Any]:
    """Convert to this project's Polyvore-outfits layout: images/, metadata.json, {train,valid,test}.json."""
    import pyarrow.parquet as pq

    from src.training.datasets import fashion_slot

    json_dir = next(p.parent for p in raw.rglob("train_no_dup.json"))
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict[str, Any]] = {}
    non_fashion: dict[str, int] = {}
    shards = sorted((raw / "images_parquet").glob("*.parquet"))
    for i, shard in enumerate(shards, 1):
        table = pq.read_table(shard, columns=["image", "category", "text", "item_ID"])
        for row in table.to_pylist():
            item_id = str(row["item_ID"])
            category = str(row["category"] or "")
            slot = fashion_slot(category)
            if slot is None:
                non_fashion[category] = non_fashion.get(category, 0) + 1
                continue
            img = row["image"] or {}
            data = img.get("bytes") if isinstance(img, dict) else None
            if not data:
                continue
            path = images / f"{item_id}.jpg"
            if not path.exists():
                from PIL import Image

                try:
                    with Image.open(io.BytesIO(data)) as im:
                        im.convert("RGB").save(path, "JPEG", quality=92)
                except Exception:
                    continue  # counted by the audit as missing
            meta[item_id] = {"semantic_category": slot, "fine_category": category, "title": row["text"]}
        print(f"  converted shard {i}/{len(shards)} ({len(meta)} fashion items so far)")

    splits = {"train": "train_no_dup.json", "valid": "valid_no_dup.json", "test": "test_no_dup.json"}
    counts: dict[str, Any] = {}
    for split, fname in splits.items():
        raw_outfits = json.loads((json_dir / fname).read_text(encoding="utf-8"))
        kept = []
        raw_items = 0
        for o in raw_outfits:
            items = [f"{o['set_id']}_{it['index']}" for it in o["items"]]
            raw_items += len(items)
            usable = [i for i in items if i in meta]
            if len(usable) >= 2:
                kept.append({"set_id": str(o["set_id"]), "items": [{"item_id": i, "index": n} for n, i in enumerate(usable, 1)]})
        (out / f"{split}.json").write_text(json.dumps(kept), encoding="utf-8")
        counts[split] = {"raw_outfits": len(raw_outfits), "raw_items": raw_items, "retained_outfits": len(kept)}
    (out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return {
        "fashion_items_with_images": len(meta),
        "non_fashion_items_dropped": sum(non_fashion.values()),
        "non_fashion_categories_dropped": dict(sorted(non_fashion.items(), key=lambda kv: -kv[1])[:40]),
        "splits": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default="maryland-polyvore", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT / "data" / "datasets")
    parser.add_argument("--yes", action="store_true", help="do not ask before large downloads")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    if args.list:
        for key, d in DATASETS.items():
            total = sum(f["size"] or 0 for f in d["files"])
            print(f"{key}: {d['name']} ({total / 2**30:.2f} GiB) — {d['license']}")
        return 0

    spec = DATASETS[args.dataset]
    base = args.root / args.dataset
    raw, prepared = base / "raw", base / "prepared"
    total = sum(f["size"] or 0 for f in spec["files"])
    free = shutil.disk_usage(args.root if args.root.exists() else args.root.parent).free
    print(f"Dataset: {spec['name']}\nLicense: {spec['license']}")
    print(
        f"Download: {total / 2**30:.2f} GiB; disk needed incl. extracted images: ~{2.5 * total / 2**30:.1f} GiB; free: {free / 2**30:.1f} GiB"
    )
    if free < 2.5 * total:
        print("error: not enough free disk space.", file=sys.stderr)
        return 2
    if total > CONFIRM_ABOVE_BYTES and not args.yes:
        if input("Continue? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 1

    manifest: dict[str, Any] = {
        "name": spec["name"],
        "dataset_key": args.dataset,
        "source": spec["source"],
        "license": spec["license"],
        "citation": spec["citation"],
        "version": spec["version"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [],
    }
    try:
        for f in spec["files"]:
            digest = download(f["url"], raw / f["path"], f["size"], f["sha256"])
            manifest["files"].append({"url": f["url"], "path": f["path"], "size": f["size"], "sha256": digest})
        extracted = safe_extract(raw / "polyvore.tar.gz", raw / "json")
        manifest["extracted_files"] = len(extracted)
        print("Preparing (decoding images, filtering non-fashion items, building splits)...")
        manifest["preparation"] = prepare_maryland(raw, prepared)
    except (DownloadError, OSError, KeyError, StopIteration) as exc:
        print(f"error: {exc}\nNothing was trained. The app keeps working with the CLIP/heuristic recommender.", file=sys.stderr)
        return 2
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Commit-worthy provenance copy (no local paths): artifacts/ is tracked, data/ is not.
    artifacts = _bootstrap.ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {base / 'manifest.json'} (copy: artifacts/dataset_manifest.json)")
    print(json.dumps(manifest["preparation"]["splits"], indent=2))

    if args.skip_audit:
        return 0
    from src.training.audit import INSUFFICIENT, DatasetSource, audit_dataset, write_reports

    print("Auditing...")
    report = audit_dataset(
        DatasetSource(
            polyvore_dir=prepared,
            metadata=prepared / "metadata.json",
            images_dir=prepared / "images",
            name=manifest["name"],
            extra={k: manifest[k] for k in ("source", "license", "version")},
        )
    )
    out = _bootstrap.ROOT / "artifacts"
    write_reports(report, out / "dataset_audit.json", out / "dataset_audit.md")
    print(f"Audit verdict: {report['gate']['verdict']}  ({out / 'dataset_audit.md'})")
    for f in report["gate"]["failures"]:
        print(f"  BLOCKING: {f}")
    return 3 if report["gate"]["verdict"] == INSUFFICIENT else 0


if __name__ == "__main__":
    sys.exit(main())
