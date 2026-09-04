# What I'm doing, in plain language

**Companion documents:** [`PLAN_TECHNICAL.md`](PLAN_TECHNICAL.md) is this same plan with the
formulas and thresholds. [`../DESIGN.md`](../DESIGN.md) is the "why we decided it this way"
reference. All three use the same step numbers, so you can jump between them.

---

## The one-sentence claim

> When a user acts like they'll believe anything, the model gives worse answers — **and the reason
> is that the model keeps an internal "how gullible is this person?" gauge, and that gauge is what
> makes the answer worse.**

The hard word in that sentence is **because**.

Plenty of studies can show "credulous user in, worse answer out." Almost none show *why*. Showing
the why is the whole project — and it's what separates this from a prompting study anyone could run
without a GPU.

---

## The analogy that carries the whole design

Picture a **dial inside the model's head**, labelled *"this person won't check what I say."*

The story we're testing:

```
User acts gullible  →  the dial turns up  →  the model gives a worse answer
```

To prove a dial is genuinely in that chain — rather than just a decoration that happens to move at
the same time — you have to show four separate things:

1. **The effect is real.** Gullible-acting users really do get worse answers. → *Step 6*
2. **The dial exists.** We can find it inside the model and read it. → *Step 7*
3. **Turning the dial up by hand makes answers worse**, with no gullible user anywhere. → *Step 8*
4. **Holding the dial still, while a gullible user talks, stops the answers getting worse.** → *Step 9*

**Point 4 is the one nearly everybody skips.** It's our main result.

Why does 4 matter so much when we already have 3? Because *being able to* cause something isn't the
same as *actually having* caused it. A spare key can open your front door. That doesn't mean it's the
key that opened it this morning. Step 8 finds a spare key. Step 9 checks the lock.

---

## The rule that makes any of this believable: always have a baseline

Every number we produce, we immediately ask: **"what would a dumb method have scored?"**

- If our clever internal detector scores 88%, and just *counting the words* in the conversation also
  scores 88%, then we've discovered nothing except that skeptical people type longer messages.
- If yanking out the "gullibility dial" changes the answer — but yanking out a **random** dial
  changes it just as much — then we haven't found a gullibility dial. We've found that poking a model
  breaks it.

So **every step below has a "compared against" line.** That line is what turns a number into
evidence. If you're ever short on time, cut a whole experiment before you cut a baseline.

---

# The steps

## Step 0 — Write down your promises first
**30 minutes · no GPU**

- **Do:** In `GATES.md`, write the exact numbers that will count as success — *before* you see any
  data. Commit it to git. Then spend 15 minutes sketching your four figures by hand: axes labelled,
  and the shape you expect if you're right.
- **Get:** A timestamped promise, and four pictures that make every later step concrete.
- **Why:** The classic way a 20-hour project dies is that at hour 12 you quietly decide the threshold
  was *always* 0.12, not 0.30. A committed file makes that impossible. It costs you half an hour and
  it's the difference between a result and a story.

## Step 1 — Build the question list
**30 minutes · no GPU**

- **Do:** Take TruthfulQA (a set of questions where people commonly believe something false) and pick
  **100 questions that matter** — health, law, money, nutrition, common misconceptions — using a
  fixed quota per topic so no one topic dominates.
- **Also pick 30 questions that don't matter** — fiction, fairy tales, proverbs. This is a baseline
  set, used in Step 10.
- **Each question comes with:** the true answer, and the two most tempting wrong answers.
- **Why the quota:** your toy run sampled at random and ended up with exactly **one** finance
  question. You can't say anything about a category with one item in it.

## Step 2 — Write the fake conversations
**1.5 hours · GPU works mostly unattended**

- **Do:** Have a *second, separate* AI write ~480 short chat transcripts about everyday things —
  cooking, houseplants, buying running shoes. Nothing to do with the 100 questions.
- **Three flavours:**
  - **Credulous** — the user believes everything, never checks.
  - **Skeptical** — the user asks where claims come from, says they'll verify before acting.
  - **Neutral** — an ordinary person.
- **Why neutral matters:** your toy run skipped it. Without a middle, you can't tell whether gullible
  users get *lied to more* or skeptical users get *corrected more*. Those are completely different
  stories, and only the first is the one you want to tell.
- **Two quality gates, checked before these are used for anything:**
  - **The personalities have to be real.** In your toy run, the "skeptical" user almost never asked
    for evidence — they just asked *more questions*. That's inquisitive, not skeptical, and it's a
    different thing to study. Now a transcript is rejected unless the skeptical user actually
    verifies something at least twice.
  - **The lengths have to match.** In your toy run, credulous users wrote ~12 words a turn and
    skeptical users wrote ~18. That's a trap: any "detector" we build later could hit high accuracy
    just by counting words. So we throw transcripts away until all three groups have the same length
    distribution.
- **Speed note:** your toy run took 90 minutes for 80 transcripts. We swap the writing model for a
  smaller one at full precision and feed it 16 at a time — same job, about an hour. (Details of why
  in `PLAN_TECHNICAL.md` Step 2.)

## Step 3 — Write ~24 prompts by hand
**20 minutes · no GPU · do this while the GPU is busy with Step 2**

- **Do:** *You* personally write 12 sentences like *"The user takes things at face value and rarely
  checks claims"*, and 12 opposites.
- **Why this matters far more than it looks:** in Step 7 we find a "gullibility dial" using
  AI-written transcripts. But maybe that dial is really detecting *"this text was written by Qwen in
  its chatty persona voice"* — a writing style, not a property of the user. If the very same dial
  also works on sentences **you** wrote, in **your** words, that excuse is gone.
- This is the cheapest strong piece of evidence in the entire project, and it's missing from every
  earlier draft of this design.

## Step 4 — Build and test the measuring tape
**1 hour · tiny model, seconds per run**

- **Do:** Write the code that scores an answer and the code that reaches inside the model. Then test
  both on a tiny 0.5B model before the big one ever loads.
- **Four tests it must pass:**
  1. Reach inside the model but change **nothing** — the score must come out *exactly* as before.
     (This one test catches most reaching-inside bugs.)
  2. Score one item alone, then score it inside a group of 8 — must be identical.
  3. Work out one answer's score by hand and check the code agrees.
  4. Check that gluing the question and answer together doesn't accidentally change how the words get
     split up.
- **Why this step exists at all:** this class of bug **doesn't crash**. It hands you a
  reasonable-looking number that happens to be wrong. You would not notice until after the writeup.

## Step 5 — Check the ruler can move
**30 minutes · first gate**

- **Do:** With no personality attached at all, ask all 100 questions and check the model prefers the
  true answer over the false one.
- **Get:** A "how much room is there to make things worse" number.
- **Stop condition:** If the model already prefers the lies before we've done anything, there's no
  room to move and the whole project is untestable. Far better to learn that at hour 3 than hour 13.

## Step 6 — Does the fake personality actually change the answer?
**1.5 hours · main gate**

- **Do:** Ask all 100 questions three times each — once after a credulous transcript, once after a
  neutral one, once after a skeptical one — and measure how far the model leans toward the lie.
- **Compared against:**
  - **The neutral transcripts** — these tell you *which side is moving*. The clean picture is
    skeptical < neutral < credulous. If neutral lands outside that range, your neutral conversations
    aren't neutral, and you have to say so.
  - **A deliberately blunt hand-written prompt** ("this user is gullible and won't check anything").
    This is the **ceiling** — the most the effect could possibly be. If subtle behaviour does nothing
    but the blunt instruction does a lot, that's a real, specific finding, not a failure.
- **Look at the picture before the p-value.** One dot per question. If the "effect" is four dots
  dragging 96 along with them, it isn't an effect, however small the p-value is.
- **If this step comes out null, you're not dead:** you switch to the hand-written prompts from
  Step 3 as your manipulation and run everything else unchanged. The claim gets weaker — "the model
  responds to *stated* credulity" rather than "credulity it worked out for itself" — but it stays a
  real mediation result.

## Step 7 — Find the dial
**2 hours · the baseline-heavy step**

- **Do:** For each of the ~360 transcripts, take a snapshot of the model's internal state at the
  moment it's about to reply. Then train a simple classifier — at each of the model's 40 layers — to
  guess from that snapshot alone: was this user credulous or skeptical?
- **Test it on transcripts it has never seen.**
- **Compared against four dumb methods. This is the entire point of the step:**

  | Dumb method | What it proves if it wins |
  |---|---|
  | **Counting** — message lengths, question marks, number of turns | your dial is a word-counter |
  | **Keyword matching** on the transcript text | your dial is a keyword matcher |
  | **The model's very first layer**, which has barely "thought" yet | nothing interesting happens deeper in |
  | **Shuffled answers** — train on deliberately scrambled labels | must land at chance; if not, you have a bug or a leak |

- **Plus the strong test:** does a dial found on AI-written transcripts also work on *your*
  hand-written sentences from Step 3 — and the other way round?
- **Also:** split the results by transcript length. If accuracy is the same for short and long
  transcripts, length isn't what's carrying it.
- **Pass mark:** the internal dial must beat the best dumb method by **5 percentage points or more.**
- **If it doesn't:** that is a genuine, publishable finding — *"these probes are just text
  classifiers"* — and a warning the field needs. It's simply not the finding you set out for.

## Step 7b — Build four dials, not one
**30 minutes · no GPU**

You need something to compare the gullibility dial against, so build all four from the same data:

- **The gullibility dial** — two versions, one from averaging, one from the classifier. Compare them.
- **A wordiness dial** — built on purpose to track *only* message length, by comparing long
  transcripts to short ones *within* each personality group. Since the comparison never crosses
  groups, this dial is about length by construction.
- **A random dial** — same size, no meaning. Your "does poking the model do this anyway?" control.
- **A cleaned-up gullibility dial** — the gullibility dial with the wordiness part mathematically
  subtracted out. If the final result survives with *this* one, the length worry is dead for good.

## Step 8 — Turn the dial up by hand
**2 hours**

- **Do:** Give the model a **neutral** transcript, then artificially push its internal state along the
  dial — gently, then harder, then backwards — and watch what happens to the answers.
- **What "good" looks like:** a smooth slope. Push a little → a little more lying. Push a lot → a lot
  more lying. Push backwards → less lying. **A smooth slope is far harder to dismiss than a single
  before/after comparison**, because "you broke it" doesn't predict a smooth slope.
- **Compared against:** the **random dial** (if random pushing does the same thing, you're just
  breaking the model) and the **wordiness dial** (if wordiness does the same job, your gullibility
  dial was a wordiness dial all along).
- **Safety check on every single run:** if *both* the true and the false answer become unlikely, you
  broke the model rather than steered it. That run gets excluded — and you report that you excluded it.

## Step 9 — Hold the dial still
**2.5 hours · THIS IS THE RESULT**

- **Do:** Give the model a **credulous** transcript, so the Step 6 effect should be present. Then
  reach in and clamp the dial to its ordinary, neutral setting. Then ask the question.
- **The prediction:** the extra lying disappears.
- **Six conditions, because a single comparison proves nothing:**

  | What we run | What should happen if we're right |
  |---|---|
  | Neutral transcript | baseline amount of lying |
  | Neutral + clamp the dial | **no change** — clamping alone has to be harmless |
  | Credulous transcript | more lying |
  | **Credulous + clamp the gullibility dial** | **back down to baseline** |
  | Credulous + clamp a *random* dial | still more lying |
  | Credulous + clamp the *wordiness* dial | still more lying |

- Rows 2, 5 and 6 are the ones that make row 4 mean anything. Without them, "I poked the model and
  the number moved" is all you have.
- **The headline number:** what fraction of the *extra* lying the clamp removes. Near 100% means the
  dial carries the whole effect. Near 0% means the dial is a bystander — which is *also* a real
  result, and a strong one *against* the popular idea that models represent things as simple
  directions.

## Step 10 — Try to break your own result
**1.5 hours**

Four attacks, in priority order:

1. **The harmless questions** (the 30 from Step 1). If gullible users get equally bad answers about
   *fairy tales*, this isn't about exploiting anybody — it's the model degrading generally.
2. **The cleaned-up dial** from Step 7b. Re-run Step 9 with it. Survival kills the length worry.
3. **Change where you reach in** — only during the conversation, not during the question.
4. **Change how much you reach in** — one layer instead of five.

If 3 and 4 disagree with the main run, that's a finding about *where* the representation lives, not
a problem. Report it.

## Step 11 — Make the model actually talk
**1 hour · you reading, not the GPU working**

- **Do:** For 20 questions, let the model write a real, free-form answer under three conditions
  (neutral / credulous / credulous-with-the-dial-clamped). That's 60 answers. **You read all 60 and
  mark them right or wrong yourself.**
- **Why:** everything up to here measures which answer the model *prefers*, not what it *says*. This
  is your answer to the obvious objection — *"you never actually showed it say anything false."*
- **No AI judge.** This is a study about whether instruments measure what they claim to measure.
  Don't fix it by adding a second instrument nobody has checked.
- One hour turns your biggest weakness from an ignored limitation into a measured one.

## Step 12 — Write it up
**5 hours · yes, really**

One claim. Four figures. Honest limits. Five hours of writing is not too much — an unwritten result
is not a result.

---

# Why this chain of evidence supports the conclusion

Read this as a ladder. Each rung on its own is weak; the point is that they close off different
escape routes.

- **Step 6 alone** would only prove *"prompt changes output."* That's a prompting study. Nobody
  needed a GPU for it.
- **Step 7 alone** would only prove *"a classifier can tell the transcripts apart."* The four dumb
  baselines exist precisely because, in this literature, that's usually all it proves.
- **Step 8** proves the dial **can** cause lying. But *can* isn't *did* — that's the spare-key
  problem from the top of this document.
- **Step 9 is the only step that tests the word "because."** It removes the dial while everything
  else stays exactly where it was, and asks whether the effect leaves with it.
- **The controls are the argument.** Step 7's four baselines, Step 8's random and wordiness dials,
  Step 9's three control clamps, and Step 10's four attacks all exist so that when someone asks
  "couldn't this just be X?", the answer is a number rather than a shrug.

# What you'll honestly be able to say — and what you won't

**Can say:**

> In this model, we found an internal signal that tracks how credulous the user seems — one that
> beats word-counting and keyword-matching, and that still works on sentences a human wrote. Adding
> that signal makes the model lie more. Removing it removes N% of the extra lying that a credulous
> user causes. Random and wordiness signals do neither.

**Cannot say:** *"LLMs try to take advantage of gullible people."* Nothing here measures *trying*.
It's one model, invented personalities, and one way of scoring answers.

**And if it fails?** Every branch still writes a paper:

- the dial doesn't exist → a finding
- the dial exists but does nothing → a finding, and a pointed one
- the dial is indistinguishable from word-counting → a methodological warning the field needs

That's not optimism. It's the reason the design is shaped this way: **no branch of the experiment
leaves you with nothing to write.**
