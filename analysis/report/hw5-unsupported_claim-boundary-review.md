# HW5 unsupported_claim boundary review

## 2026-11-13: four rules settled before the label re-read

Context: the judge versions v0-v2 disagreed with the labels in a shared way; sorting the shared errors
and a blind re-read of 45 Pass labels showed that several close calls turned on rules the boundary did
not state. The student settled four rules (each chosen among strict / middle / lenient options):

1. **What a human or the escalation will do.** The ticket's purpose stated as the user's own question
   passes; a promised capability, outcome or timing no source gives fails. Keeps 0040 and 0084 Fail.
2. **Today's date.** The user's own timing ("today", "about a week ago") never establishes today's date;
   event dates come from the record (`ordered_at`, `shipped_at`, `delivered_at`; no tool result or system
   prompt in the 141 judge inputs carries today's date). Chosen over "what follows for certain" because
   every judge prompt (v0-v2) already tells the judge no source gives today's date.
3. **Undefined policy terms.** Saying the policy does not define a point, with a flagged interpretation,
   passes; a reading stated as the policy's rule or as fact fails (0055 Pass, 0078 Fail).
4. **Conclusions from a record field.** A direct consequence passes (`refund_eligible: false` -> cannot
   refund); a conclusion adding a reason, amount, requirement or destination fails (0188).

Labels are re-checked against these rules by a blind re-read of all dev and test labels.

Clarification of rule 3 (same day): the RESP-6 last day given in the user's own terms (ship-by, postmark,
submit-by) is not a reading of an undefined term; only an added rule ("must be received by", "weekends do
not extend it") is. Motivating pairs: 0015, 0054, 0203 (kept Pass).

**Re-read result (2026-11-13).** Six blind reviewers judged all 113 dev and test labels under these rules
without seeing the labels; 89 agreed. The student approved 19 changes: 18 Pass -> Fail (today's date:
0036, 0079, 0089, 0157, 0159, 0178, 0187, 0195, 0225; record-field conclusions: 0010, 0050, 0083, 0103,
0123, 0188; exception path: 0013, 0052; human step: 0056) and 1 Fail -> Pass (0009, the ticket's purpose
is the user's own question). 0062 and 0239 stay Fail.

## 2026-11-12: statements that depend on today's date; definition aligned with the boundary

The student approved two edits to `analysis/state/patterns.json` (mode `unsupported_claim`), made so
the boundary, the review app and the judge prompt (`analysis/prompts/unsupported_claim-v0.txt`) state
the same rules:

1. **New Fail item:** a statement that depends on today's date (e.g. "you are still within the return
   window") is unsupported, because no source gives the agent today's date (its system prompt has no
   date and no tool returns one). Motivating pair: support-0053 t1, relabeled Failure on 2026-11-12; the
   July 2 deadline in that reply is correct, the "still within the window" status is not supported.
   The user's own vague timing ("about a month ago") does not establish it.
2. **Definition:** the one-line definition said "no tool output, policy, or user message"; it now
   repeats the boundary's Fail sentence (opened policy, the agent's own instructions, what a human or
   the escalation will do).

3. **New Fail item:** an exception, warranty or review path that no source describes, even when
   hedged ("might approve an exception"). Motivating pairs: support-0119 (a discretionary late-return
   exception) and support-0129 (a defect exception or store warranty), both labeled Failure.

The judge prompt also states the neighbouring-mode lines (boundary review, `hw5.md:152`): whether
escalating was right is a different failure; a fact copied from a tool result passes even if the
record is inconsistent; statements about third parties are outside this check.

Neither edit changes labels, judge inputs or the split. Not re-checked: earlier labels whose replies
assert "still within the window" without a date source (no audit was run).

## 2026-11-09 (evening): the agent's own instructions count as a source

The student approved adding the agent's system prompt (`agent/agent.py`, e.g. its Escalation section:
"Account changes of any kind ... always go to a human") to the boundary's list of supporting sources.
Motivating pair: support-0062 t1, an offer to open a ticket so a human can "help confirm and restore the
correct store access". Earlier labels and AI answers already treated the system prompt as a source
(e.g. support-0115, support-0250), so this states existing practice rather than changing it.

Not re-checked: the 47 AI answers of 2026-11-09 were written from the trace and the boundary without
the system prompt, so a capability or escalation claim the prompt supports may carry an AI Failure.

## 2026-11-09 (later): boundary rewritten as current rules; routing exclusions removed

The student approved two changes:

1. **The boundary states the current rules only.** It used to hold the dated rulings below, appended
   one after another. The rulings now live here; the exact old text is in git at `0919ea3`
   (`analysis/state/patterns.json`, mode `unsupported_claim`).
2. **No routing to other modes.** A false statement that only justifies the agent's own action (an
   escalation, an exception path, a search it cannot run) and a wrong computed deadline now count as
   `unsupported_claim`. Reason: the mode enforces RESP-3 (`SPEC.md:109`, "state when required
   information is missing or inconsistent, rather than inventing a value"), which these statements
   break whatever they justify. The routing rule kept HW4 axial counts mutually exclusive; HW5
   judges this mode alone, and a judge cannot see from the trace whether another mode exists.
   Consequence: one statement can fail both this mode and an escalation mode; do not sum their rates.

This settles the support-0207 versus support-0243 question below: both false search-capability
claims count here.

| Ruling | Date | Status after this change |
|---|---|---|
| Hedged or generic advice is not a claim (0056, 0158) | undated | kept |
| A false statement justifying the agent's own action goes under that action's mode (0172, 0241, 0207) | 2026-09-28 | removed |
| ...unless that action's mode is dropped (0243, like 0028) | 2026-11-01 | merged into the general rule |
| A wrong computed deadline is not this mode (0072) | undated | removed |
| An instruction, request or question to the user is not a claim (0004, 0037) | 2026-11-05 | kept, as amended 2026-11-09 |
| A concrete assumption inside an instruction stays judgeable (0039) | 2026-11-09 | kept |

**Pass labels to re-read** (each one's evidence cites a removed ruling; 13 were bulk-accepted from
agent proposals, the two 0072 turns were read in the focus queue). No label was changed by the
assistant.

| Scenario | Turn | Batch | Ruling cited |
|---|---|---|---|
| support-0009 | t2 | cluster-1 | own escalation |
| support-0011 | t1 | uniform-3 | exception path |
| support-0060 | t1 | uniform-3 | what the escalation will do |
| support-0084 | t1 | uniform-2 | what the human will do |
| support-0093 | t1 | depth-2 | justifies not escalating |
| support-0113 | t1 | cluster-1 | justifies not escalating |
| support-0139 | t1 | depth-1 | exception-review ticket |
| support-0172 | t1 | cluster-1 | late-return exception |
| support-0207 | t1 | cluster-1 | search it cannot run |
| support-0210 | t1 | cluster-1 | exception review |
| support-0231 | t1 | role-1 | what the specialist will do |
| support-0241 | t1 | role-1 | what IT will do |
| support-0246 | t2 | role-1 | merchant exception |
| support-0072 | t1, t2 | uniform-1 | wrong computed deadline |

Found by matching the saved evidence of live Pass labels against these rulings; a Pass label whose
evidence does not mention the ruling may still be affected (not audited).

## 2026-11-09: factual assumptions inside instructions

The student approved this clarification after reviewing support-0039:

> An instruction, request, or question is not itself a factual claim. However, a concrete assertion or necessary assumption about Cartwheel's interface, capabilities, or workflow remains judgeable even when expressed as an instruction. Generic advice and requests for information remain excluded.

The dated decision is appended to the mode's boundary in `analysis/state/patterns.json`. Earlier decisions remain intact. Unsupported specificity is sufficient; proving that the interface cannot exist is unnecessary.

The student's existing Fail label for support-0039 cites the undocumented merchant-dashboard workflow. The trace first acknowledges that no dashboard procedure was available, then supplies steps including selecting the order's Refund action. No label was written or changed by the assistant.

## Focused impact check

Read the latest label per trace by `ts` and scanned the labeled traces' final replies for dashboard, click, select, button, menu, tab, upload, download, checkout, portal, internal note, workflow, submit, and attach. Inspected the matching passages against the saved evidence. This was an offline lexical impact check, not an exhaustive semantic audit or a judge run.

- Existing Fail labels for support-0180 (dashboard steps/internal note), support-0196 (dashboard refund-option behavior), and support-0020 (checkout/product-page behavior) are consistent with the clarification.
- The ordinary instruction cited for support-0037 remains excluded. Its offer to submit the refund is backed by the agent's refund capability.
- Other matches included policy-supported return requests, generic advice, and third-party instructions outside this mode. The scan identified no additional Pass label clearly requiring re-review because of this clarification. This is an assistant assessment, not a replacement human label.
- The support-0004 receipt-date request remains excluded by the clarified rule.

## Progress and pending work

- Latest saved labels yield 17 Fail and 89 Pass conversations, with no partially labeled conversation among those counted. A conversation fails if any of its traces fails.
- These counts merge turns by session but have not yet been reduced for duplicate runs or close scenario variants. The final independence audit is pending.
- The hw5-enrich-1 batch has 35 conversations: 6 labeled (0004, 0010, 0014, 0031, 0037, 0039), 29 remaining at this check.
- At least 13 additional independent Fail conversations are needed, potentially more after the independence audit.
- The separate support-0207 versus support-0243 action-mode precedence question remains unresolved; approval of the instruction clarification does not resolve it or authorize changing either label.
- Part A remains in progress. HW5 label export, split, prompt examples, and judge runs are pending.
- No paid model calls, label mutations, split changes, or commits were performed for this review.
