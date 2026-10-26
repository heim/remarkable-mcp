"""Pure-logic builders for new reMarkable notebooks, pages, and text documents.

There is no transport / SSH here. These functions produce the bytes and JSON
dicts that the SSH write path uploads to the tablet:

- A drawable ``.rm`` page (blank, or seeded with typed text).
- A ``.content`` file describing a native notebook (the ``cPages`` page index).
- A ``.metadata`` file describing the document.
- Helpers to append a page entry to an existing notebook's ``.content``.

Pages are built with rmscene's blessed :func:`simple_text_document` builder so
every generated page contains a real drawable layer node. This is the same
mechanism validated by :mod:`remarkable_mcp.strokes` (``find_target_layer``
succeeds on the output, so strokes can be appended immediately afterwards).
``simple_text_document`` emits no ``SceneInfo`` block, so the page renders and
writes at the default paper size (see :data:`DEFAULT_PAPER`).
"""

from __future__ import annotations

import io
import re
import time
import uuid as _uuid
from typing import Optional

from rmscene import RootTextBlock, simple_text_document, write_blocks
from rmscene.crdt_sequence import CrdtId
from rmscene.scene_items import ParagraphStyle
from rmscene.tagged_block_common import LwwValue

from remarkable_mcp.strokes import WRITE_VERSION

# reMarkable 1/2 portrait stroke space. simple_text_document emits no SceneInfo,
# so render + write both fall back to this; keep them consistent.
DEFAULT_PAPER = (1404, 1872)


def new_uuid() -> str:
    """Return a fresh lowercase UUID string for a document or page id."""
    return str(_uuid.uuid4())


def _author_uuid(author_uuid: Optional[str]):
    """Coerce an author uuid (str/UUID/None) into a ``uuid.UUID``.

    A fresh author uuid is generated when none is supplied so the ``.rm``
    ``AuthorIdsBlock`` and the ``.content`` ``cPages.uuids`` entry can be kept
    consistent by the caller.
    """
    if author_uuid is None:
        return _uuid.uuid4()
    if isinstance(author_uuid, _uuid.UUID):
        return author_uuid
    return _uuid.UUID(str(author_uuid))


def page_rm_bytes(text: str = "", author_uuid: Optional[str] = None) -> bytes:
    """Return serialized ``.rm`` bytes for a single drawable page.

    With ``text=""`` this is a blank page; with text it is seeded with typed
    paragraphs (split on newlines). The result always contains a drawable layer
    node, so :func:`remarkable_mcp.strokes.append_strokes` works on it.
    """
    au = _author_uuid(author_uuid)
    blocks = list(simple_text_document(text, author_uuid=au))
    buf = io.BytesIO()
    write_blocks(buf, blocks, options={"version": WRITE_VERSION})
    return buf.getvalue()


# Markdown block markers mapped to the tablet's native paragraph styles. The
# tablet has one heading level, so "#" is Heading and deeper levels use the
# Bold paragraph style.
_CHECKBOX_RE = re.compile(r"^\s*[-*+]\s+\[([ xX])\]\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1")


def _strip_inline_markdown(line: str) -> str:
    """Drop inline markup the tablet cannot show as paragraph styles."""
    line = _LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", line)
    line = _EMPHASIS_RE.sub(lambda m: m.group(2), line)
    return line.replace("`", "")


def markdown_paragraphs(markdown: str) -> list[tuple[str, ParagraphStyle]]:
    """Split Markdown into ``(text, style)`` paragraphs for a typed-text page.

    Supports the block markup that has a native counterpart on the tablet:
    headings, bullets (two levels, by indentation), and checkboxes. Code fences
    are dropped and their lines kept as plain text. Runs of blank lines collapse
    to one, and blank lines next to a heading are dropped, since the heading
    style already has its own spacing.
    """
    paragraphs: list[tuple[str, ParagraphStyle]] = []
    in_fence = False
    for raw in markdown.replace("\r\n", "\n").split("\n"):
        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            paragraphs.append((raw, ParagraphStyle.PLAIN))
            continue

        if not raw.strip():
            if (
                paragraphs
                and paragraphs[-1][0]
                and paragraphs[-1][1]
                not in (
                    ParagraphStyle.HEADING,
                    ParagraphStyle.BOLD,
                )
            ):
                paragraphs.append(("", ParagraphStyle.PLAIN))
            continue

        if m := _HEADING_RE.match(raw):
            style = ParagraphStyle.HEADING if len(m.group(1)) == 1 else ParagraphStyle.BOLD
            text = m.group(2)
            if paragraphs and not paragraphs[-1][0]:
                paragraphs.pop()
        elif m := _CHECKBOX_RE.match(raw):
            checked = m.group(1) in "xX"
            style = ParagraphStyle.CHECKBOX_CHECKED if checked else ParagraphStyle.CHECKBOX
            text = m.group(2)
        elif m := _BULLET_RE.match(raw):
            nested = len(m.group(1).expandtabs(4)) >= 2
            style = ParagraphStyle.BULLET2 if nested else ParagraphStyle.BULLET
            text = m.group(2)
        else:
            style = ParagraphStyle.PLAIN
            text = raw.strip()
        paragraphs.append((_strip_inline_markdown(text), style))

    while paragraphs and not paragraphs[-1][0]:
        paragraphs.pop()
    return paragraphs


def markdown_page_rm_bytes(markdown: str, author_uuid: Optional[str] = None) -> bytes:
    """Return ``.rm`` bytes for a page with Markdown rendered as styled typed text.

    Builds the page like :func:`page_rm_bytes` and then sets each paragraph's
    native style. In the root text block, paragraph styles are keyed by the id
    of the newline that starts the paragraph (the first paragraph by the end
    marker ``0:0``). ``simple_text_document`` stores the text as one item with
    id ``1:16``, so character ``i`` has id ``1:16+i``.
    """
    paragraphs = markdown_paragraphs(markdown)
    text = "\n".join(p for p, _ in paragraphs)
    au = _author_uuid(author_uuid)
    blocks = list(simple_text_document(text, author_uuid=au))

    root = next(b for b in blocks if isinstance(b, RootTextBlock))
    first_item = root.value.items.sequence_items()[0]
    base = first_item.item_id
    stamp = CrdtId(1, 15)
    styles = {CrdtId(0, 0): LwwValue(timestamp=stamp, value=paragraphs[0][1])} if paragraphs else {}
    offset = 0
    for i, (para, style) in enumerate(paragraphs):
        if i > 0:
            newline_id = CrdtId(base.part1, base.part2 + offset - 1)
            styles[newline_id] = LwwValue(timestamp=stamp, value=style)
        offset += len(para) + 1
    if styles:
        root.value.styles = styles

    buf = io.BytesIO()
    write_blocks(buf, blocks, options={"version": WRITE_VERSION})
    return buf.getvalue()


def blank_page_rm_bytes(author_uuid: Optional[str] = None) -> bytes:
    """Return serialized ``.rm`` bytes for a blank drawable page."""
    return page_rm_bytes("", author_uuid=author_uuid)


def next_page_idx(existing_idx_values: list[str]) -> str:
    """Return a fractional ``idx.value`` that sorts after all existing ones.

    reMarkable orders pages by these lexicographically-sorted keys (the first
    pages of a notebook are ``ba``, ``bb``, ``bc``, ...). To append at the end
    we take the current maximum and produce the next strictly-greater key by
    incrementing its trailing character (or appending ``a`` when it is already
    ``z``, which still sorts after the original as a longer prefix-extension).
    """
    values = [v for v in existing_idx_values if isinstance(v, str) and v]
    if not values:
        return "ba"
    last = max(values)
    tail = last[-1]
    if tail < "z":
        return last[:-1] + chr(ord(tail) + 1)
    return last + "a"


def _page_entry(page_id: str, idx_value: str, template: str = "Blank") -> dict:
    """Build a single ``cPages.pages[]`` entry."""
    return {
        "id": page_id,
        "idx": {"timestamp": "1:2", "value": idx_value},
        "template": {"timestamp": "1:2", "value": template},
    }


def new_notebook_content(
    page_ids: list[str],
    author_uuid: str,
    paper: tuple[int, int] = DEFAULT_PAPER,
) -> dict:
    """Build a ``.content`` dict for a brand-new native notebook.

    Mirrors the schema xochitl itself writes (``formatVersion`` 2, ``cPages``
    page index, portrait zoom defaults). ``author_uuid`` must match the uuid
    used to build the page ``.rm`` bytes so CRDT author ids line up.
    """
    width, height = paper
    pages = []
    idx_values: list[str] = []
    for page_id in page_ids:
        idx_value = next_page_idx(idx_values)
        idx_values.append(idx_value)
        pages.append(_page_entry(page_id, idx_value))

    return {
        "cPages": {
            "lastOpened": {"timestamp": "1:1", "value": page_ids[0] if page_ids else ""},
            "original": {"timestamp": "0:0", "value": -1},
            "pages": pages,
            "uuids": [{"first": str(author_uuid), "second": 1}],
        },
        "coverPageNumber": -1,
        "customZoomCenterX": 0,
        "customZoomCenterY": height // 2,
        "customZoomOrientation": "portrait",
        "customZoomPageHeight": height,
        "customZoomPageWidth": width,
        "customZoomScale": 1,
        "documentMetadata": {},
        "extraMetadata": {},
        "fileType": "notebook",
        "fontName": "",
        "formatVersion": 2,
        "lineHeight": -1,
        "margins": 125,
        "orientation": "portrait",
        "pageCount": len(page_ids),
        "pageTags": [],
        "sizeInBytes": "0",
        "tags": [],
        "textAlignment": "justify",
        "textScale": 1,
        "zoomMode": "bestFit",
    }


def new_document_metadata(visible_name: str, parent: str = "") -> dict:
    """Build a ``.metadata`` dict for a new document.

    Includes the sync flags (``metadatamodified``/``modified``/``synced``/
    ``version``) so a freshly created document is picked up and synced, matching
    the proven ``remarkable_upload`` SSH path.
    """
    now = str(int(time.time() * 1000))
    return {
        "visibleName": visible_name,
        "type": "DocumentType",
        "parent": parent or "",
        "createdTime": now,
        "lastModified": now,
        "lastOpened": now,
        "lastOpenedPage": 0,
        "pinned": False,
        "deleted": False,
        "metadatamodified": True,
        "modified": True,
        "synced": False,
        "version": 0,
    }


def append_page_to_content(content_data: dict, new_page_id: str) -> dict:
    """Append a blank page entry to an existing notebook's ``.content`` dict.

    Returns ``{"content": <updated dict>, "idx": <new idx value>,
    "page_index": <1-based index>, "total_pages": <new count>}``. The input
    dict is updated in place. Raises :class:`ValueError` if the document is not
    a native ``cPages`` notebook (e.g. a PDF/EPUB with a flat ``pages`` list).
    """
    cpages = content_data.get("cPages")
    if not isinstance(cpages, dict) or not isinstance(cpages.get("pages"), list):
        raise ValueError(
            "Document is not a native notebook (no cPages page index); "
            "pages can only be added to notebooks."
        )

    pages = cpages["pages"]
    existing_idx = [
        p.get("idx", {}).get("value")
        for p in pages
        if isinstance(p, dict) and isinstance(p.get("idx"), dict)
    ]
    idx_value = next_page_idx([v for v in existing_idx if isinstance(v, str)])
    pages.append(_page_entry(new_page_id, idx_value))

    total = len(pages)
    content_data["pageCount"] = total
    return {
        "content": content_data,
        "idx": idx_value,
        "page_index": total,
        "total_pages": total,
    }
