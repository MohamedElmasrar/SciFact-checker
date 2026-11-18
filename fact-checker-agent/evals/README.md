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

## Scripted comparison

The baseline ran against revision `fd2f0df` with the same fixtures and abstract
search. The updated run enabled bounded full-text lookup and quote validation.

| Metric | Baseline | Updated |
| --- | ---: | ---: |
| Correct verdicts, all 32 cases | 25 | 31 |
| Correct verdicts, 16 holdout cases | 13 | 15 |
| Wrong definitive verdicts | 2 | 0 |
| Correct abstentions | 8 | 10 |
| Correct definitive verdicts, out of 22 possible | 17 | 21 |
| Definitive verdicts with text-matched quotes | 0 | 21 |
| Local fixture p95 | 13 ms | 20 ms |

The updated run abstained on a supported claim after the fixture supplied a
fabricated quote. Fixture latency measures local graph and tool overhead.
Live Gemini quality and provider latency require a separate run.

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
