# TakeMeter — Planning Document

**Community:** r/nba  
**Project:** Fine-tuned discourse quality classifier  
**Model:** distilbert-base-uncased fine-tuned on labeled r/nba posts

---

## Community

**What community did you choose and why?**

I chose r/nba (the NBA subreddit, ~8 million members). It is one of the most active sports
communities on Reddit, generating thousands of posts and comments daily. Discourse quality
varies enormously: some posts make structured statistical arguments about player performance
or team strategy; others are confident assertions stated without any evidence; others are
pure emotional reactions to live game moments. This variation makes the community an ideal
fit for a discourse-quality classifier — the distinctions are real, community members
recognize and debate them explicitly, and the text volume makes it easy to collect 200
balanced examples.

The NBA community also uses specific vocabulary and framing conventions that a fine-tuned
model can learn (e.g., citation of PER/WS/TS%, game-log references, "the eye test" as a
rhetorical move), making the classification task genuinely interesting and non-trivial for
a zero-shot baseline.

---

## Labels

Three labels capture the meaningful spectrum of discourse quality in r/nba:

---

### `analysis`

**Definition:** A post that makes a structured argument backed by specific, verifiable
evidence — statistics, historical comparisons, play-by-play observations, or tactical
reasoning. The claim is supported, not just asserted. Removing the opinion framing would
still leave a recognizable argument.

**Example 1:**
> "People sleep on Jokic's defense. His DRPM this season is +1.8, which ranks in the top
> 15 among centers. He gives up fewer points in the paint than most traditional rim
> protectors because he angles drives instead of contesting vertically. The film clearly
> shows this on pick-and-roll coverage."

**Example 2:**
> "The Celtics' net rating in clutch situations (last 5 min, margin ≤5) is +12.3 this
> season, best in the league. Their shot quality in those moments — measured by average
> shot distance and frequency of corner 3s — is dramatically better than their regular
> game profile. This is not luck; it's scheme."

---

### `hot_take`

**Definition:** A bold, confident opinion stated without supporting evidence. The claim
might be true or false, but the post asserts rather than argues. Evidence, if present,
is decorative or cherry-picked to sound credible rather than genuinely reasoning.

**Example 1:**
> "LeBron is cooked. I don't care what anyone says. He was never a closer and at 39 he
> absolutely cannot guard anyone in the playoffs. The Lakers are a first-round exit and
> it's entirely on him."

**Example 2:**
> "Wembanyama is already better than Giannis. Giannis never had that handle at 20.
> Book it. This guy wins 3 MVPs before he turns 27."

---

### `reaction`

**Definition:** An immediate emotional response to a specific game event, play, or
moment. Little to no argument — the post is expressing a feeling in real time. Typically
short, uses game-specific language, and provides no analysis or opinion beyond the
immediate response.

**Example 1:**
> "STEPH CURRY FROM HALF COURT AT THE BUZZER WHAT IS THIS MAN"

**Example 2:**
> "I can't believe we just blew a 20-point lead to the Pistons. I'm done. Pack it up.
> Season over."

---

## Hard Edge Cases

**The most genuinely ambiguous boundary: `hot_take` vs `analysis`**

Some posts include one specific statistic alongside a bold framing. Example:
> "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."

This could be `hot_take` (accusatory framing, single cherry-picked stat) or `analysis`
(cites a specific verifiable fact).

**Decision rule:** If the post provides specific, verifiable evidence that would support
the claim even if you removed the opinion framing, label it `analysis`. If the evidence
is vague, cherry-picked, or used decoratively — just enough to sound credible but not
genuinely reasoning — label it `hot_take`. One stat selected for rhetorical effect, with
no comparison or context, is `hot_take`.

**The second ambiguous boundary: `hot_take` vs `reaction`**

A strong opinion about a game outcome could be either. Example:
> "That was the worst coaching decision I've ever seen. We deserved to lose."

**Decision rule:** If the post is tied to a specific live moment (in-game, just happened)
and focuses on the event rather than making a broader claim, label `reaction`. If the post
makes a general claim about a player, team, or trend — even in emotional language — label
`hot_take`.

---

## Data Collection Plan

**Source:** Reddit public posts and comments from r/nba. Collection via manual
copy-paste from the subreddit web interface (sorted by Hot, New, and Top filters
to get variety). No private content or authenticated-only content.

**Target distribution:**
- `analysis`: 70 examples (35%)
- `hot_take`: 70 examples (35%)
- `reaction`: 60 examples (30%)

Total: 200 examples minimum.

**If a label is underrepresented after 200 examples:** Sort r/nba by "New" during
a live game window to find more `reaction` posts; use the search filter
"site:reddit.com/r/nba statistics" to surface more `analysis` candidates.

**Split:** Single labeled CSV uploaded to the Colab notebook, which handles the
70/15/15 train/validation/test split automatically.

---

## Evaluation Metrics

**Primary metric: macro-averaged F1**

Overall accuracy alone is not sufficient because the dataset has a slight class
imbalance (reaction is smaller) and because the cost of misclassification is not
uniform — a `hot_take` classified as `analysis` is more misleading than a
`reaction` classified as `hot_take`. Macro F1 weights each class equally regardless
of size, which forces the model to perform well across all three categories.

**Secondary metrics:**
- Per-class precision and recall: to understand which specific boundaries the model
  has learned and which it confuses.
- Confusion matrix: to identify directional error patterns (e.g., does the model
  systematically predict `hot_take` when the truth is `analysis`?).

**Baseline comparison:**
- Zero-shot baseline: Groq `llama-3.3-70b-versatile` prompted with label definitions,
  run on the same locked test set.
- The fine-tuned model should meaningfully exceed the baseline on macro F1. If it
  does not, this is a signal that labels are too noisy or the training set is too small.

---

## Definition of Success

**Minimum bar for deployment utility:**
- Fine-tuned model overall accuracy ≥ 70% on the test set
- No per-class F1 below 0.55 (a model that collapses one class is not useful)
- Fine-tuned model outperforms zero-shot baseline by at least 10 percentage points
  on overall accuracy

**Ideal performance:**
- All per-class F1 ≥ 0.70
- Macro F1 ≥ 0.72
- Confusion matrix shows no single off-diagonal cell above 30% of its row total

If the fine-tuned model barely beats the baseline, the most likely causes are
annotation inconsistency (especially on the `hot_take`/`analysis` boundary) or
insufficient examples of the hard edge cases in the training set.

---

## AI Tool Plan

### Label stress-testing
Before annotating 200 examples, I will give Claude the three label definitions and the
hard edge case description above, and ask it to generate 8–10 posts that sit on the
`hot_take`/`analysis` boundary. If I cannot classify them cleanly using my decision rule,
I will tighten the rule before committing to annotation.

### Annotation assistance
I will use an LLM (Claude or Groq) to pre-label a batch of ~50 examples by providing
the label definitions and raw post text. I will review and correct every pre-assigned
label before including it in the dataset. I will disclose which examples were pre-labeled
in the AI Usage section of my README. Pre-labeled examples that I cannot confidently
verify will be discarded rather than included with uncertain labels.

### Failure analysis
After running evaluation, I will paste the misclassified examples into Claude and ask it
to identify common themes — post length, use of sarcasm, label pairs that keep getting
confused, presence of statistics without context. I will verify the patterns myself by
re-reading the examples before writing up the evaluation report. I will note which
patterns I confirmed and which I discarded.
