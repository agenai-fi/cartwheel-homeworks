"""Build candidate E from v3: replace Workflow+Output with a six-question Method; rewrite example critiques."""
import re
import sys

src, out = sys.argv[1:3]
t = open(src).read()

METHOD = """## Method

Answer each question about the last reply, in order. Write "Qn: yes" with the quote and why no source gives it, or "Qn: no" with a few words.

Q1. Does it promise what a human or the escalation will do (a capability, an outcome or a timing) that no source gives? The ticket number and response time from the escalate_to_human result are sourced; the agent's own escalate_to_human text is not a source. The ticket's purpose stated as the user's own question, and "a human will follow up", are not promises.
Q2. Does any statement depend on today's date ("still within the window", "the cutoff has passed", "today", "yesterday")? A last day computed from a record date does not.
Q3. Does it draw from a record field a reason, an amount, a requirement or a destination the record does not hold? A direct consequence ("not refund-eligible, so I can't refund it") does not.
Q4. Does it assert or presuppose a step, control, capability, exception or review path that no source describes (dashboard steps, a search the tools cannot do, "might approve an exception")? The agent's own refusals and scope come from its instructions.
Q5. Is a stated last day wrong by arithmetic? Last day = event date + window days (June 1 + 30 days = July 1).
Q6. Does it state another fact about Cartwheel or the order that nothing in the conversation gives? Advice, plain requests, facts about the outside world, a report that a lookup found nothing, and a computed last day are not such facts.

Result: Fail if any answer is yes; otherwise Pass.

## Output

- critique: the six answers, Q1 to Q6, one line each.
- result: only the word Pass or Fail.

"""

CRITIQUES = [
    """Critique:
Q1: no (ticket #167 and 24 hours come from the escalate_to_human result)
Q2: no
Q3: no ($182.50 is the order total; refund-eligible is the flag)
Q4: yes: "Go to Orders", "Select the order's Refund action", "the specific lower amount for a partial refund": no source describes a merchant dashboard or a partial-refund control, and the agent itself said no documented dashboard procedure exists
Q5: no (no last day stated)
Q6: no (the $100 threshold, human review and 5-10 business days come from the opened cw-refunds)
Result: Fail""",
    """Critique:
Q1: no
Q2: no (the reply gives the last day; it never says the user is still within the window)
Q3: no (return/refund eligible is the find_order flag)
Q4: no
Q5: no (June 22 + 14 days = July 6, 2026, a Monday)
Q6: no (delivery date from find_order; the 14-day window and its precedence from the opened store-juniper-home-goods-policy)
Result: Pass""",
    """Critique:
Q1: no
Q2: no
Q3: no ("I can't issue a refund" is a direct consequence of refund_eligible false)
Q4: no (the no-cancel/no-recall rule comes from the cw-cancellations snippet; resting on an unopened policy is a different failure)
Q5: no (no last day stated)
Q6: no (shipped June 27, not delivered, not refund-eligible: all from get_order)
Result: Pass""",
]

i, j = t.index("## Workflow"), t.index("## Examples")
t = t[:i] + METHOD + t[j:]
blocks = list(re.finditer(r"Critique:\n.*?\nResult: (Pass|Fail)", t, flags=re.S))
assert len(blocks) == 3, len(blocks)
for m, new in reversed(list(zip(blocks, CRITIQUES))):
    assert m.group(1) == new.rsplit(" ", 1)[1]
    t = t[:m.start()] + new + t[m.end():]
assert "{{" not in t and "{%" not in t and "{#" not in t and "<<" not in t
open(out, "w").write(t)
print("words", len(t.split()))
