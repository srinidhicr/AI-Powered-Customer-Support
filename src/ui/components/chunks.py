# src/ui/components/chunks.py
"""Renders source-chunk data as an expandable HTML block for gr.HTML."""

from __future__ import annotations


def render(chunks: list) -> str:
    """
    Convert a list of source-chunk dicts into a self-contained HTML
    <details> block that Gradio's gr.HTML component can display inline.

    Returns an empty string when chunks is empty so the gr.HTML component
    stays invisible.
    """
    if not chunks:
        return ""

    rows: list[str] = []
    for i, ch in enumerate(chunks):
        subj  = ch.get("subject") or f"Chunk {i + 1}"
        score = ch.get("score", 0)
        body  = ch.get("answer", "")[:300]
        tags  = ch.get("tags", [])

        tag_html = " ".join(
            f'<span style="font-size:10px;padding:1px 6px;border-radius:3px;'
            f'background:#1e2433;color:#4b5680;border:1px solid #252a38">{t}</span>'
            for t in tags[:4]
        )

        rows.append(
            f'<div style="padding:10px 14px;border-bottom:1px solid #1e2433">'
            f'  <div style="display:flex;justify-content:space-between;'
            f'       align-items:flex-start;margin-bottom:4px">'
            f'    <span style="font-weight:500;font-size:12px;color:#c8d0e8;flex:1">'
            f'      {subj[:72]}'
            f'    </span>'
            f'    <span style="font-family:monospace;font-size:10px;'
            f'          background:rgba(91,122,245,.13);color:#5b7af5;'
            f'          padding:1px 7px;border-radius:3px;margin-left:8px;white-space:nowrap">'
            f'      score {score:.3f}'
            f'    </span>'
            f'  </div>'
            f'  <div style="font-size:12px;color:#7b849e;line-height:1.5">{body}</div>'
            + (f'  <div style="margin-top:5px">{tag_html}</div>' if tag_html else "")
            + "</div>"
        )

    n     = len(chunks)
    label = f"📄 {n} source chunk{'s' if n > 1 else ''} used — click to view"

    return (
        f'<details style="margin-top:6px">'
        f'  <summary style="cursor:pointer;font-size:11px;color:#5b7af5;user-select:none;'
        f'    list-style:none;display:inline-flex;align-items:center;gap:5px;'
        f'    padding:3px 10px;border:1px solid #252a38;border-radius:5px;'
        f'    background:transparent">'
        f'    {label}'
        f'  </summary>'
        f'  <div style="margin-top:6px;background:#0f1218;border:1px solid #1e2433;'
        f'       border-radius:8px;overflow:hidden;max-width:100%">'
        + "".join(rows)
        + "  </div>"
        f"</details>"
    )