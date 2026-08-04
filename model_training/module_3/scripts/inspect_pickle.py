"""
Run locally:
    python inspect_pickle2.py processed/models/vedavision_species_model.pkl

Tracks unicode string pushes so we can reconstruct STACK_GLOBAL
(module, qualname) pairs -- this is how modern pickle protocols (4/5)
encode class/function references, instead of the older GLOBAL opcode.
"""
import pickletools
import sys

path = sys.argv[1]

with open(path, "rb") as f:
    data = f.read()

print(f"File size: {len(data)} bytes\n")

# Track the last two string-pushing ops seen, since STACK_GLOBAL
# consumes the two strings pushed immediately before it (module, name).
recent_strings = []
refs = []

STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
    "SHORT_BINSTRING", "BINSTRING",
}

for opcode, arg, pos in pickletools.genops(data):
    if opcode.name in STRING_OPS:
        recent_strings.append(arg)
        # keep only last 2
        recent_strings = recent_strings[-2:]
    elif opcode.name == "GLOBAL":
        # arg is "module name" as one string for GLOBAL
        refs.append(tuple(arg.split(" ", 1)) if isinstance(arg, str) else (arg,))
    elif opcode.name == "STACK_GLOBAL":
        if len(recent_strings) >= 2:
            refs.append((recent_strings[-2], recent_strings[-1]))
        else:
            refs.append(("<unknown>", "<unknown>"))

print(f"Total references found: {len(refs)}\n")

# de-dupe, keep order
seen = set()
unique_refs = []
for r in refs:
    if r not in seen:
        seen.add(r)
        unique_refs.append(r)

for module, name in unique_refs:
    flag = "  <-- SUSPECT" if module == name or module == "VotingClassifier" else ""
    print(f"  module={module!r:50s} name={name!r}{flag}")