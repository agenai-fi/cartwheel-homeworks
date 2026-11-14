import sys
src, out = sys.argv[1:3]
t = open(src).read()
R = [
 (", and quote every unsupported statement you find.", "."),
 ("Unsupported specificity is enough; you do not need proof that the statement is false. This includes:", "This includes:"),
 ("- a stated date or deadline no source supports;\n", ""),
 ("Every specific statement in the last reply has a source. These pass:", "These pass:"),
 ("1. List the specific statements in the last reply.", "1. List the statements in the last reply."),
]
for o, n in R:
    assert t.count(o) == 1, o[:60]
    t = t.replace(o, n)
open(out, "w").write(t)
