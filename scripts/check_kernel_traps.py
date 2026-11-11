#!/usr/bin/env python3
"""Checks that every arch-guarded CUDA kernel body stops on the targets it has no code for.

src/cuda/README.md's rule: one binary holds every target architecture, so a kernel body that
uses newer instructions sits behind an `#if` on `__CUDA_ARCH__` (or a macro defined under one,
such as `LEGION_FP4_MMA`), and ends in `#else __trap(); #endif` (or ggml's `NO_DEVICE_CODE`).
Without the `#else`, older targets compile an empty kernel: a launch the host picked wrongly
leaves its output unwritten and reports success.

For each `__global__` definition under the given directories (default: src/cuda), the script
finds the preprocessor groups at the top level of its body whose condition names the
architecture (or a macro defined under such a condition, also through other such macros). A
group with no `#else` that holds code while the rest of the body does no work and no other such
group has an `#else` with work, or one whose `#else` does no work, is an offender: printed as
`file:line: kernel name` (the `#if` line), and the script exits with status 1. Statements that
do no work on their own count as no code: declarations and `static_assert`s (setup), `return;`
and `if (...) return;`, `__syncthreads();`, `(void) x;`, calls of the `pdl_*` helpers (they
wait or trigger only) and block delimiters or control headers on their own. Work under a nested
arch `#if` counts only when all its branches, an `#else` included, do work; inside a group
whose own condition does not name the architecture, such a guard with a gap is checked like a
top-level one. `__device__` helpers are not checked: they may keep an older fallback or
be no-ops elsewhere, and the kernels calling them carry the guard. No compiler and no GPU needed
(a CI step).

Usage: scripts/check_kernel_traps.py [dir-or-file ...]
"""

import re
import sys
from pathlib import Path

ARCH_RE = re.compile(r"__CUDA_ARCH")
DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif|define)\b(.*)$")
# Statements that do nothing on their own: programmatic dependent launch's wait and trigger,
# barriers, early returns and casts to void. A guarded body with only these (and declarations)
# outside it is still empty elsewhere, and an `#else` with only these does not stop the kernel.
NO_WORK_RE = re.compile(
    r"^(pdl_\w+\s*\(\s*\)|__syncthreads\s*\(\s*\)|return|if\s*\(.*\)\s*return"
    r"|\(\s*void\s*\)\s*\w+)\s*;$"
)
# A declaration (setup: a variable, constant or type that a guarded body may read), with or
# without an initializer: qualifiers, then a type (template arguments, pointers and their
# qualifiers allowed), then names with optional array bounds. `x = y;`, `out[i] = v;` and
# calls do not match.
DECL_RE = re.compile(
    r"^(static_assert\b.*|(using|typedef)\b.*;|"
    r"((const|constexpr|static|volatile|extern|__shared__|__align__\s*\(\s*\w+\s*\)|"
    r"unsigned|signed|long|short)\s+)*"
    r"[A-Za-z_][\w:]*(\s*<[^;=]*>)?(?=[\s*&])(\s|\*|&|const\b|volatile\b|__restrict__\b)+"
    r"[A-Za-z_]\w*(\s*\[[^\]]*\])*(\s*,\s*\**\s*[A-Za-z_]\w*(\s*\[[^\]]*\])*)*\s*(=[^=].*)?;)$"
)


# Block delimiters and control headers (`{`, `}`, `} else {`, `if (x) {`): the statements inside
# the block decide whether it does work.
BLOCK_RE = re.compile(r"^(\}\s*)*([^;{}]*\{\s*)?(\}\s*)*$")


def does_work(line):
    """Whether a code line (stripped) is more than setup, a block delimiter or a statement that
    does no work."""
    if BLOCK_RE.match(line):
        return False
    # A block on one line (`{ return; }`): its statement decides.
    inner = line.strip("{} \t")
    return bool(inner) and not (NO_WORK_RE.match(inner) or DECL_RE.match(inner))


def strip_comments(text):
    """`text` with comments, string and character literals blanked (newlines kept), so braces
    and directives inside them do not count."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(c + " " * (j - i - 2) + (c if j - i >= 2 else ""))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def arch_macros(files):
    """Macros `#define`d only under an architecture condition (LEGION_FP4_MMA and the like),
    including a condition on another such macro, or with a value that names the architecture or
    such a macro: a condition on them is a condition on the architecture. Collected until no pass adds one (a macro may be defined before or in
    another file than the one it depends on)."""
    names = set()
    while True:
        before = len(names)
        for text in files.values():
            stack = []
            for line in text.splitlines():
                m = DIRECTIVE_RE.match(line)
                if not m:
                    continue
                kind, rest = m.group(1), m.group(2)
                if kind in ("if", "ifdef", "ifndef"):
                    stack.append(is_arch(rest, names))
                elif kind == "elif" and stack:
                    stack[-1] = stack[-1] or is_arch(rest, names)
                elif kind == "endif" and stack:
                    stack.pop()
                elif kind == "define":
                    # Under an arch condition, or with a value that names the architecture
                    # (`#define USE_SM80 (__CUDA_ARCH__ >= 800)`).
                    name = rest.split()[0] if rest.split() else ""
                    value = rest.strip()[len(name):]
                    name = name.split("(")[0]
                    if name and (any(stack) or is_arch(value, names)):
                        names.add(name)
        if len(names) == before:
            return names


def is_arch(cond, macros):
    if ARCH_RE.search(cond):
        return True
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cond))
    return bool(words & macros)


def kernels(text):
    """(line of `__global__`, name, first body line, last body line) per kernel definition,
    lines 0-based; `text` without comments."""
    lines_at = [0]
    for i, c in enumerate(text):
        if c == "\n":
            lines_at.append(i + 1)

    def line_of(pos):
        lo, hi = 0, len(lines_at) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if lines_at[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    for m in re.finditer(r"\b__global__\b", text):
        # The signature ends at the body's `{` or a declaration's `;` (parentheses balanced:
        # `__launch_bounds__(...)` and the parameters).
        i, depth = m.end(), 0
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif depth == 0 and c in "{;":
                break
            i += 1
        if i >= len(text) or text[i] == ";":
            continue
        sig = text[m.end():i]
        names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", sig)
        names = [x for x in names if x not in ("__launch_bounds__", "__cluster_dims__", "alignas")]
        name = names[0] if names else "?"
        open_at, depth = i, 0
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield line_of(m.start()), name, line_of(open_at), line_of(i)


def offenders(text, macros):
    """(line, name) of each arch-guarded kernel body that compiles empty on other targets."""
    lines = text.splitlines()
    found = []
    for at, name, first, last in kernels(text):
        body = lines[first:last + 1]
        if body:
            # Code on the `{` line after the brace, and on the `}` line before it.
            body[0] = body[0].split("{", 1)[1] if "{" in body[0] else ""
            if first != last:
                body[-1] = body[-1].rsplit("}", 1)[0] if "}" in body[-1] else body[-1]
        # The body's top-level groups, and its statements outside them that do some work. `stack`
        # holds one level per open `#if`: whether its condition names the architecture, whether
        # it reached `#else`, and per branch whether that branch does work. A nested level's
        # work reaches the branch around it when it runs on every target: a plain `#if` with
        # work, or an arch `#if` whose branches, `#else` included, all do work (a write under
        # `#if __CUDA_ARCH__ >= 800` inside an sm_90 group's `#else` leaves sm_70 empty).
        groups, stack = [], []
        outside = 0
        # Statements are judged once they end (or a directive cuts them); directives flush the
        # pending one, so the stack is the same for all its lines.
        pending = []

        def flush():
            nonlocal outside, pending
            if not pending:
                return
            work = does_work(" ".join(pending))
            pending = []
            if not stack:
                outside += work
            else:
                groups[-1]["code"] += 1
                stack[-1]["branches"][-1] = stack[-1]["branches"][-1] or work

        for k, line in enumerate(body):
            m = DIRECTIVE_RE.match(line)
            if m:
                flush()
                kind, rest = m.group(1), m.group(2)
                if kind in ("if", "ifdef", "ifndef"):
                    if not stack:
                        groups.append({
                            "line": first + k,
                            "arch": False,
                            "else": False,
                            "else_code": False,
                            "code": 0,
                            "nested_gap": False,
                            "plain_work": False,
                        })
                    stack.append({"arch": is_arch(rest, macros), "else": False, "branches": [False]})
                    if len(stack) == 1:
                        groups[-1]["arch"] = stack[0]["arch"]
                elif kind == "elif" and stack:
                    stack[-1]["arch"] = stack[-1]["arch"] or is_arch(rest, macros)
                    stack[-1]["branches"].append(False)
                    if len(stack) == 1:
                        groups[-1]["arch"] = stack[0]["arch"]
                elif kind == "else" and stack:
                    stack[-1]["else"] = True
                    stack[-1]["branches"].append(False)
                    if len(stack) == 1:
                        groups[-1]["else"] = True
                elif kind == "endif" and stack:
                    lvl = stack.pop()
                    if not stack:
                        groups[-1]["else_code"] = lvl["else"] and lvl["branches"][-1]
                        groups[-1]["plain_work"] = any(lvl["branches"])
                    else:
                        if lvl["arch"]:
                            work = lvl["else"] and all(lvl["branches"])
                            # An arch guard inside a group whose own condition does not name
                            # the architecture (`#ifdef FOO` around `#if __CUDA_ARCH__ >= 800`).
                            groups[-1]["nested_gap"] |= not work
                        else:
                            work = any(lvl["branches"])
                        stack[-1]["branches"][-1] = stack[-1]["branches"][-1] or work
                continue
            code = line.strip()
            if not code or code.startswith("#"):
                continue
            pending.append(code)
            if code.endswith((";", "{", "}")):
                flush()
        flush()
        # Another arch group's `#else` that works (a trap or a fallback) runs on the targets the
        # group without one leaves out (an sm_90 hint before an sm_80 body that traps).
        covered = any(g["arch"] and g["else_code"] for g in groups)
        for g in groups:
            if not g["arch"]:
                # Its only work under a nested arch guard that leaves some targets empty.
                if g["nested_gap"] and not g["plain_work"] and outside == 0 and not covered:
                    found.append((g["line"], name))
                continue
            if g["else"] and not g["else_code"]:
                found.append((g["line"], name))
            elif not g["else"] and outside == 0 and not covered and g["code"] > 0:
                found.append((g["line"], name))
    return found


def main(args):
    roots = [Path(a) for a in args] or [Path(__file__).resolve().parent.parent / "src" / "cuda"]
    paths = []
    for r in roots:
        if r.is_dir():
            paths += sorted(p for p in r.rglob("*") if p.suffix in (".cu", ".cuh"))
        else:
            paths.append(r)
    files = {p: strip_comments(p.read_text()) for p in paths}
    macros = arch_macros(files)
    bad = 0
    for p, text in files.items():
        for line, name in offenders(text, macros):
            print(f"{p}:{line + 1}: kernel {name}: an arch-guarded body without "
                  "`#else __trap();` (an empty kernel on the other targets)")
            bad += 1
    if bad:
        print(f"{bad} kernel(s) without a trap stub; see src/cuda/README.md", file=sys.stderr)
        return 1
    n = sum(1 for t in files.values() for _ in kernels(t))
    print(f"{n} kernels in {len(files)} files: every arch-guarded body traps elsewhere")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
