#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import libcst as cst

REPO = Path(__file__).resolve().parents[1]

_PRAGMA = re.compile(
    r"#\s*(noqa|type:\s*ignore|pyright:|pragma:)\b",
    re.IGNORECASE,
)

def _keep_trailing_comment(text: str) -> bool:
    return bool(_PRAGMA.search(text))

def _is_docstring_stmt(node: cst.CSTNode) -> bool:
    if not isinstance(node, cst.SimpleStatementLine):
        return False
    if len(node.body) != 1:
        return False
    stmt = node.body[0]
    if not isinstance(stmt, cst.Expr):
        return False
    return isinstance(stmt.value, (cst.SimpleString, cst.ConcatenatedString))

def _strip_leading_lines(lines: tuple[cst.EmptyLine, ...]) -> tuple[cst.EmptyLine, ...]:
    out: list[cst.EmptyLine] = []
    for el in lines:
        c = el.comment
        if c is None:
            out.append(el)
            continue
        if _keep_trailing_comment(c.value):
            out.append(el)
            continue
        out.append(cst.EmptyLine(whitespace=el.whitespace))
    return tuple(out)

class StripCommentsAndDocstrings(cst.CSTTransformer):
    def leave_TrailingWhitespace(
        self, original_node: cst.TrailingWhitespace, updated_node: cst.TrailingWhitespace
    ) -> cst.TrailingWhitespace:
        c = updated_node.comment
        if c is None:
            return updated_node
        if _keep_trailing_comment(c.value):
            return updated_node
        return updated_node.with_changes(comment=None)

    def leave_EmptyLine(
        self, original_node: cst.EmptyLine, updated_node: cst.EmptyLine
    ) -> cst.EmptyLine:
        c = updated_node.comment
        if c is None:
            return updated_node
        if _keep_trailing_comment(c.value):
            return updated_node
        return updated_node.with_changes(comment=None)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        body = list(updated_node.body)
        if body and _is_docstring_stmt(body[0]):
            body = body[1:]
        return updated_node.with_changes(
            header=_strip_leading_lines(updated_node.header),
            body=tuple(body),
        )

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        b = updated_node.body
        if not isinstance(b, cst.IndentedBlock):
            return updated_node
        stmts = list(b.body)
        if stmts and _is_docstring_stmt(stmts[0]):
            stmts = stmts[1:]
        return updated_node.with_changes(
            body=b.with_changes(
                body=tuple(stmts),
                footer=_strip_leading_lines(b.footer),
            )
        )

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        b = updated_node.body
        if not isinstance(b, cst.IndentedBlock):
            return updated_node
        stmts = list(b.body)
        if stmts and _is_docstring_stmt(stmts[0]):
            stmts = stmts[1:]
        return updated_node.with_changes(
            body=b.with_changes(
                body=tuple(stmts),
                footer=_strip_leading_lines(b.footer),
            )
        )

def _strip_tex_full_line_comments(path: Path) -> None:
    raw = path.read_text()
    lines = raw.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if line.lstrip().startswith("%"):
            continue
        out.append(line)
    path.write_text("".join(out))

def _iter_py() -> list[Path]:
    paths: list[Path] = []
    for root in (REPO / "standard_qa", REPO / "realistic_qa"):
        if root.is_dir():
            paths.extend(sorted(root.rglob("*.py")))
    mp = REPO / "modal_runner.py"
    if mp.is_file():
        paths.append(mp)
    return paths

def main() -> int:
    xf = StripCommentsAndDocstrings()
    for path in _iter_py():
        src = path.read_text(encoding="utf-8")
        try:
            tree = cst.parse_module(src)
        except cst.ParserSyntaxError as e:
            print(f"skip parse {path}: {e}", file=sys.stderr)
            continue
        new_tree = tree.visit(xf)
        new_src = new_tree.code
        if new_src != src:
            path.write_text(new_src, encoding="utf-8")
            print(path)
    for name in ("table_3way_standard_v4.tex", "table_3way_standard_v5.tex", "table_delta_quality_sig.tex"):
        p = REPO / name
        if p.is_file():
            _strip_tex_full_line_comments(p)
            print(p)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
