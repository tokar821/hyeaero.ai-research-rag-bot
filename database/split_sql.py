"""Split migration SQL into statements (semicolons outside single-quoted strings only)."""

from __future__ import annotations


def sql_statements(sql_text: str) -> list[str]:
    lines_out: list[str] = []
    for line in sql_text.splitlines():
        if line.strip().startswith("--"):
            continue
        lines_out.append(line)
    blob = "\n".join(lines_out)
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    i = 0
    n = len(blob)
    while i < n:
        c = blob[i]
        if c == "'":
            if in_quote and i + 1 < n and blob[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_quote = not in_quote
            buf.append(c)
            i += 1
            continue
        if c == ";" and not in_quote:
            stmt = "".join(buf).strip()
            if stmt:
                parts.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts
