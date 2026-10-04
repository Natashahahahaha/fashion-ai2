"""Downloader safety: path traversal, checksums, resume (local HTTP server, no internet)."""

from __future__ import annotations

import hashlib
import importlib
import io
import sys
import tarfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
dl = importlib.import_module("download_dataset")


class RangeHandler(SimpleHTTPRequestHandler):
    """Static server with basic Range support (enough to test resuming)."""

    def log_message(self, *args):  # keep test output quiet
        pass

    def send_head(self):
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().send_head()
        start = int(rng.split("=")[1].split("-")[0])
        data = path.read_bytes()
        self.send_response(206)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        return io.BytesIO(data[start:])


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    httpd = HTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(root)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_verifies_size_and_checksum(server, tmp_path):
    root, url = server
    payload = b"fashion" * 1000
    (root / "f.bin").write_bytes(payload)
    good = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "out" / "f.bin"
    assert dl.download(f"{url}/f.bin", dest, len(payload), good) == good
    assert dest.read_bytes() == payload
    with pytest.raises(dl.DownloadError, match="SHA-256 mismatch"):
        dl.download(f"{url}/f.bin", tmp_path / "out" / "g.bin", len(payload), "0" * 64)
    assert not (tmp_path / "out" / "g.bin").exists()
    with pytest.raises(dl.DownloadError, match="expected"):
        dl.download(f"{url}/f.bin", tmp_path / "out" / "h.bin", len(payload) + 5, None)


def test_download_resumes_partial_file(server, tmp_path):
    root, url = server
    payload = bytes(range(256)) * 400
    (root / "big.bin").write_bytes(payload)
    dest = tmp_path / "big.bin"
    dest.with_name("big.bin.part").write_bytes(payload[:10000])  # interrupted earlier
    digest = dl.download(f"{url}/big.bin", dest, len(payload), hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload and digest == hashlib.sha256(payload).hexdigest()


def test_unreachable_url_fails_readably(tmp_path):
    with pytest.raises(dl.DownloadError, match="re-run"):
        dl.download("http://127.0.0.1:9/nothing", tmp_path / "x.bin", None, None, retries=1)


def _tar_with(tmp_path: Path, name: str, link: bool = False) -> Path:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        else:
            data = b"x"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


@pytest.mark.parametrize("name,link", [("../escape.txt", False), ("/abs/path.txt", False), ("link", True)])
def test_safe_extract_refuses_unsafe_members(tmp_path, name, link):
    archive = _tar_with(tmp_path, name, link)
    with pytest.raises(dl.DownloadError, match="unsafe"):
        dl.safe_extract(archive, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_normal_archive(tmp_path):
    archive = _tar_with(tmp_path, "polyvore/train_no_dup.json")
    assert dl.safe_extract(archive, tmp_path / "dest") == ["polyvore/train_no_dup.json"]
    assert (tmp_path / "dest" / "polyvore" / "train_no_dup.json").is_file()


def test_fashion_slot_mapping():
    from src.training.datasets import fashion_slot

    assert fashion_slot("Day Dresses") == "one_piece"
    assert fashion_slot("Skinny Jeans") == "bottom"
    assert fashion_slot("Ankle Booties") == "shoes"
    assert fashion_slot("Blazers") == "outerwear"
    assert fashion_slot("Earrings") == "accessory"
    assert fashion_slot("Floral Decor") is None  # not an outfit item
    assert fashion_slot("Clothing") is None  # ambiguous slot
    assert fashion_slot("Lipstick") is None
