"""Live GenEvolve tools with the upstream skill bank and bounded transport."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import ipaddress
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urljoin, urlsplit

from .common import atomic_write

MAX_IMAGE_BYTES = 16 << 20
MAX_PIXELS = 589824  # Pinned LLaMA-Factory default.


class InfrastructureError(RuntimeError):
    """Only sanitized categories are written to progress receipts."""


def serper_configuration():
    key = os.environ.get("SERPER_API_KEY", "").strip()
    base = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev").rstrip("/")
    url = urlsplit(base)
    if not key or "\n" in key or "\r" in key:
        raise InfrastructureError("missing_search_api_configuration")
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise InfrastructureError("invalid_search_api_endpoint")
    return base, key


def public_http_url(value):
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
        raise ValueError("Not a public HTTP image URL")
    if url.port not in (None, 80, 443):
        raise ValueError("Unexpected image URL port")
    addresses = socket.getaddrinfo(url.hostname, url.port or (443 if url.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError("Non-public image endpoint")
    return value


def clean_field(value):
    # Tool payloads cannot introduce fake image placeholders.
    return str(value or "").replace("<image>", "[image]").replace("\r", " ").replace("\n", " ")


class LiveTools:
    def __init__(self, out, *, search_rps):
        from genevolve.knowledge_tool import KnowledgeTool
        from genevolve.tools.web_search import WebTextSearchTool
        self.knowledge = KnowledgeTool()
        self.text_format = WebTextSearchTool._format_text_results
        self.base, self.key = serper_configuration()
        self.out = out
        self.interval = 1.0 / search_rps
        self.next_request = 0.0
        self.lock = threading.Lock()
        self.images = ThreadPoolExecutor(max_workers=16)

    def close(self):
        self.images.shutdown(wait=True, cancel_futures=True)

    def post(self, endpoint, query, k):
        import requests
        for attempt in range(3):
            with self.lock:
                now = time.monotonic()
                delay = max(0.0, self.next_request - now)
                self.next_request = now + delay + self.interval
            if delay:
                time.sleep(delay)
            reason = "search_transport_failure"
            try:
                reply = requests.post(self.base + "/" + endpoint,
                    headers={"X-API-KEY": self.key, "Content-Type": "application/json"},
                    json={"q": query, "num": k}, timeout=(8, 30), allow_redirects=False)
                if reply.status_code in (401, 403):
                    raise InfrastructureError("search_auth_failure")
                if reply.status_code == 429:
                    reason = "search_rate_or_quota_failure"
                elif 500 <= reply.status_code < 600:
                    reason = "search_service_failure"
                elif reply.status_code != 200:
                    raise InfrastructureError("search_http_failure")
                else:
                    data = reply.json()
                    field = "organic" if endpoint == "search" else "images"
                    if (not isinstance(data, dict) or
                            (field not in data and "searchParameters" not in data)):
                        raise InfrastructureError("invalid_search_service_response")
                    found = data.get(field) or []
                    if not isinstance(found, list) or any(not isinstance(x, dict) for x in found):
                        raise InfrastructureError("invalid_search_service_response")
                    return found[:k]
            except requests.RequestException:
                pass
            except ValueError:
                reason = "invalid_search_service_response"
            if attempt < 2:
                time.sleep(1 + 2 * attempt)
        raise InfrastructureError(reason)

    def preflight(self):
        self.post("search", "Eiffel Tower Paris architecture", 1)
        self.post("images", "Eiffel Tower Paris architecture", 1)

    def _download(self, url):
        import requests
        from PIL import Image, ImageOps
        try:
            # Validate each redirect. The image request never carries the search API key.
            for _ in range(4):
                with requests.get(public_http_url(url), timeout=(5, 12), stream=True,
                                  allow_redirects=False) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers["Location"])
                        continue
                    if response.status_code != 200:
                        return None
                    raw = bytearray()
                    for piece in response.iter_content(chunk_size=1 << 16):
                        raw.extend(piece)
                        if len(raw) > MAX_IMAGE_BYTES:
                            return None
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.width * opened.height > 40_000_000:
                        return None
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                if image.width * image.height > MAX_PIXELS:
                    scale = (MAX_PIXELS / (image.width * image.height)) ** 0.5
                    image.thumbnail((max(1, int(scale * image.width)), max(1, int(scale * image.height))),
                                    Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=90)
                content = buf.getvalue()
                sha = hashlib.sha256(content).hexdigest()
                name = f"images/{sha}.jpg"
                if not (self.out / name).exists():
                    atomic_write(self.out / name, lambda f: f.write(content))
                return name
        except (requests.RequestException, OSError, ValueError, KeyError, Image.DecompressionBombError):
            return None
        return None

    def execute(self, name, args, image_manager, max_results):
        """Return the textual observation, ordered visual assets, and a compact status."""
        try:
            k = min(max(1, int(args.get("top_k") or max_results)), max_results)
        except (ValueError, TypeError, OverflowError):
            return "Invalid top_k.", [], "invalid_arguments"
        if name == "search":
            queries = args.get("queries") or []
            if isinstance(queries, str):
                queries = [queries]
            if (not isinstance(queries, list) or not 0 < len(queries) <= 8
                    or any(not isinstance(q, str) or not q.strip() or len(q) > 1024 for q in queries)):
                return "Provide a nonempty list of short search queries.", [], "invalid_arguments"
            blocks = []
            for q in queries:
                entries = [{key: clean_field(v) for key, v in row.items()}
                           for row in self.post("search", q, k)]
                blocks.append(self.text_format(clean_field(q), entries, k))
            return "\n".join(blocks), [], "ok"
        if name == "image_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip() or len(query) > 1024:
                return "Provide a nonempty image-search query.", [], "invalid_arguments"
            hits = self.post("images", query, k)
            items = []
            for h in hits:
                url = h.get("imageUrl") or h.get("image_url") or h.get("thumbnailUrl") or h.get("url")
                if isinstance(url, str) and url:
                    items.append({
                        "url": url, "title": clean_field(h.get("title") or "image"),
                        "page_url": clean_field(h.get("link") or h.get("source")),
                    })
            for item, future in zip(items, [self.images.submit(self._download, x["url"]) for x in items]):
                item["local_path"] = future.result()
            refs = image_manager.register([x for x in items if x["local_path"]])
            lines = [f"--- image search result for [{clean_field(query)}] ---"]
            for ref in refs:
                lines += [f"{ref['img_id']}: <image>", "  title: " + ref["title"],
                          "  url: " + clean_field(ref["url"]), "  local_path: " + ref["local_path"]]
                if ref["page_url"]:
                    lines.append("  page_url: " + ref["page_url"])
            lines.append("--- end of image search result ---")
            return "\n".join(lines), [r["local_path"] for r in refs], (
                "partial_or_unavailable_images" if len(refs) < len(hits) else "ok"
            )
        if name == "query_knowledge":
            skill = args.get("skill_name")
            if not isinstance(skill, str) or not skill.strip():
                return "Provide a skill_name.", [], "invalid_arguments"
            return self.knowledge.call(skill_name=skill).replace("<image>", "[image]"), [], "ok"
        return f"Unknown tool: {clean_field(name)}", [], "unknown_tool"
