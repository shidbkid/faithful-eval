# TODO

- Persist per-example `preds` in results JSON and replace Hanley–McNeil AUC CIs
  with bootstrap CIs (decision gate softener until then).
- Judge scaling curve: Qwen 0.5B / 1.5B / 3B / 7B-q vs NLI horizontal line.
- Cascade: ROUGE → NLI → judge on uncertain cases.
- Failure complementarity once preds are saved.
- Third dataset only after the Medium post is up.
