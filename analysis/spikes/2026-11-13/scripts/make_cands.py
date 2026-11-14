import sys
src, outA, outB = sys.argv[1:4]
t = open(src).read()
a_old = '- the agent\'s instructions (the system prompt at the start), e.g. "account changes always go to a human".'
a_new = ('- the agent\'s instructions (the system prompt at the start), including what it helps with and what it refuses, e.g. "account changes always go to a human";\n'
         '- arithmetic on dates and windows from these sources (a computed last day);\n'
         '- an empty or null tool result, for a statement that something was not found or is not recorded.')
b_old = '1. List the specific statements in the last reply.'
b_new = ('1. List the statements in the last reply that assert something checkable about Cartwheel, the order, the agent\'s abilities '
         'or what a human will do. Skip advice, plain requests and facts about the outside world, unless they assume a step, control, capability or path.')
assert t.count(a_old) == 1 and t.count(b_old) == 1
A = t.replace(a_old, a_new)
open(outA, "w").write(A)
open(outB, "w").write(A.replace(b_old, b_new))
