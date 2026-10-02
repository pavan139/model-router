# Publishing a routing experiment

The useful unit to share is a frozen decision rule plus the records on which it was evaluated. Keep `policies/response_fixture.jsonl`, the router artifact, evaluation JSON, sensitivity runs, and estimator notes with the code. The fixture response costs are configured integer units, not invoices.

Run `python tools/check_routing.py --require-lint --wheel` before creating a release. Check the source digest and test/wheel outcomes in `reports/verification.json`. A release of this directory activates its own GitHub Actions workflow and needs no collection-level imports.

For imported model responses, retain exact source-model identity and timezone-qualified capture times. Omit private response text rather than inventing provenance. For logged-bandit results, preserve action probabilities and only the chosen outcome in estimator input. State whether intervals are conditional finite-sample IPS bounds or asymptotic Wald intervals.

Create the remote repository under your account, then push this directory's branch. Keep provider credentials, downloaded weights, environments, and scratch output out of the commit. The measured synthetic result can be published as an offline routing experiment; no live provider savings have been measured here.
