# TakeMeter — r/nba Discourse Quality Classifier

A fine-tuned text classifier that evaluates discourse quality in r/nba posts.
Built for AI201 Week 3 Project.

---

## Community Choice

**Community:** r/nba (NBA subreddit, ~8 million members)

r/nba generates thousands of posts daily with enormous variation in discourse quality —
from rigorous statistical arguments to pure emotional reactions. The distinctions are
real and community members debate them explicitly, making it an ideal classification task.
The community also uses specific vocabulary (PER, TS%, net rating, etc.) that a fine-tuned
model can learn, creating a non-trivial boundary for the baseline to struggle with.

---

## Label Taxonomy

Three labels capture the meaningful spectrum of r/nba discourse:

### `analysis`
A post that makes a structured argument backed by specific, verifiable evidence —
statistics, historical comparisons, play-by-play observations, or tactical reasoning.
Removing the opinion framing still leaves a recognizable argument.

- **Example 1:** "Over the last 12 game 7s, 10 have been decided by double digits.
  Half by 20+. Average margin of victory: 19.5. The 'best two words in sports' myth
  doesn't hold up to the data."
- **Example 2:** "Conference Finals starting lineups by net rating: Indiana +22.1,
  Minnesota +19.1, Thunder +3.9, Knicks -4.7. Indiana and OKC had the two most
  efficient starting lineups in the regular season."

### `hot_take`
A bold, confident opinion stated without supporting evidence. The post asserts rather
than argues. Any evidence present is decorative, not genuine reasoning.

- **Example 1:** "Can Jalen Brunson be the face of the NBA if the Knicks win a
  championship? With LeBron, Steph and KD in their twilight, the league needs a new
  face — Brunson fits the bill."
- **Example 2:** "Should the Nuggets trade Murray for Giannis? A Giannis-Jokic pairing
  would be like Robinson-Duncan or Shaq-Malone but healthier."

### `reaction`
An immediate emotional response to a specific game event or play. Little to no argument —
the post expresses a feeling in real time.

- **Example 1:** "Just finished watching OKC/DEN and I am left with one question:
  is hand checking legal now? I was not rooting for either team but that was obvious."
- **Example 2:** "What is going on with defense in NBA? I have the feeling defense is
  not clean often — the lack of consistency makes me appreciate the games less."

---

## Data Collection

**Source:** Reddit public posts and comments from r/nba, collected via the PullPush.io
API (a public Reddit data mirror). Posts were captured from the 2024-25 NBA playoff
window (May 2025), a period with high discourse density.

**Collection method:** Automated script pulling submissions and top-level comments.
430 raw items collected; filtered to 200 after removing moderator announcements, pure
link posts, and non-English content.

**Labeling process:** Posts were pre-labeled using Groq `llama-3.3-70b-versatile`
with the label definitions and edge-case rules from planning.md. Every pre-assigned
label was reviewed by the annotator before inclusion. Pre-labeling is disclosed in the
AI Usage section below.

**Label distribution:**

| Label | Count | % |
|-------|-------|---|
| analysis | 87 | 43.5% |
| hot_take | 47 | 23.5% |
| reaction | 66 | 33.0% |
| **Total** | **200** | **100%** |

**Three difficult-to-label examples:**

1. *"LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."*
   Could be `hot_take` (accusatory framing) or `analysis` (cites a specific stat).
   **Decision:** `hot_take` — one cherry-picked stat used for rhetorical effect, no
   broader comparison or context.

2. *"The Thunder open as -375 favourites against the Wolves. Seems a bit too much lol.
   They are also 7.5 point favourites in game 1."*
   Could be `analysis` (cites betting odds) or `reaction` (live playoff moment).
   **Decision:** `hot_take` — cites odds but the core claim ("seems too much") is
   an unsupported opinion about market pricing.

3. *"What Is going on with defense in NBA? I have the feeling defense is not clean often."*
   Could be `reaction` (responding to a game) or `hot_take` (broader opinion).
   **Decision:** `reaction` — post is tied to watching a specific game in real time,
   focused on in-the-moment frustration rather than a sustained argument.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:** Google Colab free T4 GPU. The Colab notebook handles the
train/validation/test split automatically (70% / 15% / 15% = 140 / 30 / 30 examples).

**Key hyperparameters:**
- Epochs: 3 (default)
- Learning rate: 2e-5 (default)
- Batch size: 16 (default)

No hyperparameter changes were made from the notebook defaults, as the training
loss appeared to converge within 3 epochs on 140 examples.

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no fine-tuning)

**Prompt used:**
```
You are labeling r/nba Reddit posts for a discourse quality classifier.
Assign exactly one label: analysis, hot_take, or reaction

analysis = structured argument with specific verifiable stats/evidence.
hot_take = bold confident opinion with no real evidence.
reaction = immediate emotional response to a specific game moment.

[Edge-case decision rules included]

Reply with ONLY the label word.
```

**Baseline results were collected on the locked test set (30 examples) before
fine-tuning began.**

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Baseline (Groq zero-shot) | **80.0%** (24/30) |
| Fine-tuned (DistilBERT) | **43.3%** (13/30) |

### Per-Class Metrics

**Baseline:**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 0.76 | 1.00 | 0.87 | 13 |
| hot_take | 0.75 | 0.86 | 0.80 | 7 |
| reaction | 1.00 | 0.50 | 0.67 | 10 |
| **macro avg** | **0.84** | **0.79** | **0.78** | 30 |

**Fine-tuned (DistilBERT):**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 0.43 | 1.00 | 0.60 | 13 |
| hot_take | 0.00 | 0.00 | 0.00 | 7 |
| reaction | 0.00 | 0.00 | 0.00 | 10 |
| **macro avg** | **0.14** | **0.33** | **0.20** | 30 |

### Confusion Matrix (Fine-Tuned Model)

| | Pred: analysis | Pred: hot_take | Pred: reaction |
|---|---|---|---|
| True: analysis | **13** | 0 | 0 |
| True: hot_take | 7 | 0 | 0 |
| True: reaction | 10 | 0 | 0 |

### Wrong Prediction Analysis

The fine-tuned model predicted **all 30 test examples as `analysis`**, correctly
classifying only the 13 true analysis examples. All 17 errors follow the same pattern.

**Wrong prediction 1:**
> "Can Jalen Brunson be the face of the NBA if the Knicks win a championship? With the
> careers of LeBron, Steph and KD in their twilight, there's a lot of talk about the
> next face of the league."

- True label: `hot_take` | Predicted: `analysis`
- **Why it failed:** This post is framed as a question with implicit reasoning ("careers
  in twilight"), which superficially resembles an analytical structure. The model likely
  learned surface-level patterns (citing players, referencing careers) rather than
  whether evidence genuinely supports a claim.

**Wrong prediction 2:**
> "Should the Nuggets trade Murray for Giannis? A Giannis and Jokic pairing would be
> brilliant — like when the Spurs had Robinson and Duncan."

- True label: `hot_take` | Predicted: `analysis`
- **Why it failed:** Historical comparisons (Robinson-Duncan, Shaq-Malone) are present,
  which are surface signals the model associated with `analysis`. But these comparisons
  are asserted without data, not used to build an argument.

**Wrong prediction 3:**
> "So, is hand checking legal now? Just finished watching OKC/DEN and I am left with
> one question — it was obvious to me that OKC fouled the ball handler on almost
> every possession."

- True label: `reaction` | Predicted: `analysis`
- **Why it failed:** The post contains game-specific observations that look like
  tactical analysis ("fouled the ball handler on almost every possession"), but the
  intent is emotional frustration at officiating. The model cannot distinguish
  observation-based reaction from genuine tactical breakdown.

### Sample Classifications (Fine-Tuned Model)

| Post (truncated) | True Label | Predicted | Confidence |
|---|---|---|---|
| "Over the last 12 game 7s, 10 decided by double digits..." | analysis | analysis | ~0.93 |
| "Conference Finals lineups by net rating: Indiana +22.1..." | analysis | analysis | ~0.91 |
| "Can Jalen Brunson be the face of the NBA..." | hot_take | **analysis** ❌ | ~0.72 |
| "Is hand checking legal now?..." | reaction | **analysis** ❌ | ~0.68 |
| "SGA laughing after game 3 loss — coldest moment ever?" | hot_take | **analysis** ❌ | ~0.70 |

*Note: Confidence scores are approximate softmax outputs from the fine-tuned model.*

---

## Reflection: What the Model Learned vs. What I Intended

I intended the model to learn the structural distinction between evidence-backed reasoning,
unsupported assertion, and in-the-moment emotional response.

What the model actually learned was a simpler rule: **predict `analysis` for everything**.
This happened because `analysis` was the majority class (87/200 = 43.5%) in the training
data, and with only 140 training examples, the model found it more efficient to memorize
the dominant class than to learn the three-way boundary.

The gap between intent and outcome reveals two problems:

1. **Class imbalance:** 87 analysis vs 47 hot_take examples is a 1.85:1 ratio. With 200
   total examples, the minority class had only ~33 training examples — too few to learn
   a reliable boundary.

2. **Surface feature overlap:** Many `hot_take` posts cite statistics or historical
   comparisons as rhetorical devices. The model learned "contains numbers/comparisons =
   analysis" rather than "evidence genuinely supports the claim."

The baseline (Groq zero-shot) outperforms the fine-tuned model because it has the semantic
understanding to distinguish decorative statistics from genuine reasoning — a capability
that requires far more than 140 labeled examples to learn from scratch.

---

## Spec Reflection

**One way the spec helped:** The hard edge case section in planning.md (specifically the
`hot_take` vs `analysis` decision rule) forced me to define the boundary before annotating.
Without that explicit rule — "one stat selected for rhetorical effect = hot_take" — I
would have labeled the borderline posts inconsistently, producing even noisier training data.

**One way implementation diverged from the spec:** The spec targeted 70/70/60 label
distribution. The actual distribution ended up 87/47/66. The `hot_take` class was
underrepresented because genuine unsupported-assertion posts are less common in r/nba
during playoff season, when discourse trends toward analysis and reaction. I should have
collected more targeted data (e.g., off-season trade speculation threads) to balance this.

---

## AI Usage

1. **LLM pre-labeling:** Groq `llama-3.3-70b-versatile` was used to pre-label all 200
   examples using the label definitions and edge-case rules from planning.md. Every
   pre-assigned label was reviewed by the annotator; approximately 15-20% were corrected,
   particularly on the `hot_take`/`analysis` boundary. Corrected labels reflect the
   annotator's final judgment.

2. **Failure pattern analysis:** After evaluation, misclassified examples were analyzed
   with Claude to identify common themes. The AI identified the majority-class collapse
   pattern and the surface-feature overlap between `hot_take` and `analysis`. Both
   patterns were confirmed by re-reading the examples manually before inclusion in this
   report.
