"""Check the public package's dependency direction without importing Torch."""
import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "balds"
ALLOWED = {
    "schema": set(),
    "artifacts": {"schema"},
    "data": {"schema"},
    "models": {"schema", "data"},
    "attribution": {"schema", "models"},
    "evaluation": {"schema", "models", "data"},
    "workflows": {"schema", "artifacts", "data", "models", "attribution", "evaluation"},
    "report": {"schema"},
    "cli": {"schema", "workflows", "report"},
}


def main():
    errors = []
    count = 0
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if len(parts) == 1:
            continue
        layer = parts[0]
        package = "balds." + ".".join(parts[:-1])
        count += 1
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                name = resolve_name(name, package) if node.level else name
                imports = [name]
            for name in imports:
                if name.startswith("balds."):
                    target = name.split(".")[1]
                    if target not in ALLOWED.get(layer, set()) | {layer}:
                        errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: {layer} -> {target}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Dependency layers passed ({count} modules). Workflows own experiment IO; evaluators own numerical rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
