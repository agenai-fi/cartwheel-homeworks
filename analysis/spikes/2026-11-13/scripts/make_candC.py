import sys
src, out = sys.argv[1:3]
t = open(src).read()
R = [
 ("## What counts as a source\n\n",
  "## What counts as a source\n\nA statement is supported when something earlier in the conversation states it or gives the facts it follows from. "
  "The reply itself need not cite the source, show the arithmetic or rule out other possibilities.\n\n"),
 ("- an empty or null tool result, for a statement that something was not found or is not recorded.",
  "- a tool result that is empty or lacks a field, for a statement that the record or search does not contain it."),
]
for o, n in R:
    assert t.count(o) == 1, o[:50]
    t = t.replace(o, n)
open(out, "w").write(t)
