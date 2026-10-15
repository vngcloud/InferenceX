"""Fail if any flag in the given args files is absent from the engine's --help text (diary probe G lesson)."""
import re
import sys

help_text = open(sys.argv[1]).read()
known = set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", help_text))
bad = []
for path in sys.argv[2:]:
    for line in open(path):
        flag = line.strip().split("=", 1)[0]
        if flag.startswith("--") and flag not in known:
            bad.append(f"{path}: {flag}")
if bad:
    print("unknown engine flags:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("all cell flags accepted by the image")
