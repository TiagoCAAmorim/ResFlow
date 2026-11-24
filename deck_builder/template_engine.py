"""Tiny template engine for the IMEX deck template (templates/spe10_2d.tpl).

  {name}              replaced by the value of `name` in the current context
  #for NAME ... #end  the enclosed lines are written once per item of the list ctx[NAME]; each item is a
                      dict whose keys become variables inside the block (loops can be nested; an item may
                      hold a list for a nested loop)
  #if NAME ... #end   the enclosed lines are written only if ctx[NAME] is truthy
Directive lines start with '#' (IMEX comments start with '**', so there is no clash). A name that is not
defined raises KeyError, so a typo never goes unnoticed.
"""
import re

_VAR = re.compile(r"\{(\w+)\}")


def _parse(lines):
    root, stack = [], []
    cur = root
    for n, line in enumerate(lines, 1):
        tok = line.strip().split()
        if tok and tok[0] in ("#for", "#if"):
            if len(tok) != 2:
                raise ValueError(f"line {n}: expected '{tok[0]} NAME'")
            node = (tok[0][1:], tok[1], [])
            cur.append(node)
            stack.append(cur)
            cur = node[2]
        elif tok and tok[0] == "#end":
            if not stack:
                raise ValueError(f"line {n}: '#end' without '#for'/'#if'")
            cur = stack.pop()
        else:
            cur.append(line)
    if stack:
        raise ValueError("unclosed '#for'/'#if' block")
    return root


def _render(nodes, ctx, out):
    for node in nodes:
        if isinstance(node, str):
            out.append(_VAR.sub(lambda m: str(ctx[m.group(1)]), node))
        elif node[0] == "for":
            for item in ctx[node[1]]:
                _render(node[2], {**ctx, **item}, out)
        elif ctx[node[1]]:
            _render(node[2], ctx, out)


def render(template, ctx):
    out = []
    _render(_parse(template.splitlines()), ctx, out)
    return "\n".join(out) + "\n"
