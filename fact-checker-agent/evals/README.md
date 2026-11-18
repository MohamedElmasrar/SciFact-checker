# Verification evaluation

`claims.jsonl` contains 32 synthetic study claims with frozen paper abstracts,
expected verdicts, and grading rationales. The development and holdout splits
contain 16 cases each. The suite covers direct support, contradiction, conflicting
results, missing evidence, scope mismatches, and conclusions that require PDF
passages. It isolates verification behavior from live search relevance.

Run from `fact-checker-agent`:

```bash
uv run --locked python evals/run.py --check --output /tmp/fact-check-report.json
uv run --locked python evals/run.py --split holdout --check
```

The scripted mode exercises the real verification graph with deterministic model
responses and frozen MCP tool results. The grader checks the verdict, the paper
identities on decisive citations, and whether each decisive quote matches its
source passage after whitespace normalization. The report includes each claim's
rationale, explanation, citations, and elapsed time. CI checks for zero wrong
definitive verdicts and execution errors, requires all definitive quotes and
citations to pass, and allows one abstention on a claim with decisive evidence.

## Run with Gemini

With a configured Gemini key in the environment or agent's `.env`, run:

```bash
uv run --locked python evals/run.py --mode gemini --split holdout \
  --output /tmp/fact-check-live-holdout.json
```

This sends the selected synthetic claims and retrieved fixture passages to the
configured Gemini model and can incur API charges. Review semantic support for
each borderline case against the supplied passages. A matching quote and paper
identity establish traceability. Whether the passage actually entails the claim
still requires judgment.

## Published-paper pilot

`published-claims.json` contains 20 scoped claims across five papers. The
development split uses the Transformer, ResNet, and LoRA papers. The holdout
split uses the calibration and GW150914 papers, with each paper confined to one
split. Each case identifies the exact PDF, physical page, section, a short
excerpt, and the reason the nearby evidence does or does not establish the claim.

### Controlled assessment

Configure a Gemini key and run this command from `fact-checker-agent`.

```bash
GEMINI_MODEL=gemini-3.1-flash-lite uv run --locked python evals/run.py \
  --suite published --mode gemini --evidence controlled --requests-per-minute 10 \
  --output /tmp/fact-check-published-controlled.json
```

The runner sends each original claim and one pinned PDF passage through the
production assessment prompt and citation guard. It skips claim extraction and
live retrieval. The model receives the passage, paper title, and page. The
expected verdict and grading rationale are available only to the scorer. Each
case makes one Gemini request. The passages deliberately supply the decisive
evidence for supported and contradicted cases. The four abstention cases supply
related text that leaves the original claim unresolved.

### End-to-end assessment

Index SciFact using the [workspace setup guide](../../README.md#run-locally).
Run this command from `fact-checker-agent`. Stop the dev server if it is
using the same local index.

```bash
GEMINI_MODEL=gemini-3.1-flash-lite uv run --locked python evals/run.py \
  --suite published --mode gemini --evidence live --requests-per-minute 10 \
  --output /tmp/fact-check-published-live.json
```

The production graph receives the original claim and retrieves evidence normally.
Each case makes up to three Gemini requests. Both modes share one request pacer
across their cases. When running both modes back-to-back, wait at least one
request interval between commands to maintain the cap across processes. Use
`--split holdout`, `--limit 2`, or `--case calibration-02` for a smaller run.

### Reading the reports

- `correct` counts automatic passes. A decisive pass requires the original
  claim, the expected verdict, grounded quotes, and a quote matching the
  verified reference excerpt. Whitespace is normalized when comparing claims.
  Reworded claims and valid text quotes outside the pinned excerpt require
  semantic review. Inspect the interpretation, quote, and explanation before
  crediting a paraphrase or alternate evidence. `verdict_matches_label` is the
  raw label count, including partial results.
- The retrieval totals distinguish paper metadata, passages from the reference
  paper, and passages containing the pinned excerpt. Excerpt matching is
  literal after whitespace normalization and may miss equivalent evidence.
  The per-call report captures the query and the actual assessment input,
  including passage text, IDs, and PDF page numbers when available.
- Each Gemini call records the pacing wait, time in the client, structured
  output when available, and a specific outcome. The aggregation distinguishes
  extraction mismatches, schema errors, HTTP responses, and cancellations.
  A failed full-text follow-up preserves the initial assessment and marks
  the case as partial. Partial results and failed requests stay unscored.
- Keep `scored`, `review_required`, and `unscored` denominators visible when
  comparing runs.
