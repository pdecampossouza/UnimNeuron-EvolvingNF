# Team Contribution Log

This document records the individual contributions of team members to the Data Stream Mining Research Project at NOVA IMS.

Its purpose is to ensure transparency, accountability and recognition of individual work.

## Team Members

| Member | GitHub Username |
|---|---|
| Bruno Henriques | @brunophenriques |
| Diogo Padilha | @diogonpadilha |
| João Pinho | @pinho-1 |
| Maria Inês | @mariainessantos |
| Luis Soeiro | @luisrafaelsoeiro |

## Contribution History

| Date | Member | Contribution | Evidence |
|---|---|---|---|
| 2026-11-09 | Bruno Henriques | Set up the team collaboration workflow, including branching strategy, commit conventions, Pull Request guidelines, and contribution tracking. | [PR #1](https://github.com/brunophenriques/T1DSM/pull/1) [PR #2](https://github.com/brunophenriques/T1DSM/pull/2) |
| 2026-11-19 | Maria Inês | Implemented the feature-relevance module on branch `mines`: read-only extraction and checkpoint logging of the model's global and per-rule feature weights, tie-aware stability and validity metrics, a synthetic stream with ground-truth relevance and drift, and a config- and seed-driven runner. Added tests for non-intrusive logging, absence of label leakage, and equivalence with the baseline prequential loop. Developed with AI assistance (Claude); code reviewed and tested locally. | [relevance/](https://github.com/brunophenriques/T1DSM/tree/mines/relevance) [run_relevance.py](https://github.com/brunophenriques/T1DSM/blob/mines/run_relevance.py) [configs/](https://github.com/brunophenriques/T1DSM/tree/mines/configs) [tests/](https://github.com/brunophenriques/T1DSM/tree/mines/tests) |

## Guidelines

- Update this log whenever a meaningful task is completed.
- Include contributions involving research, coding, experiments, analysis, documentation and writing.
- Provide supporting evidence whenever possible.
- Do not record planned tasks as completed contributions.
- All team members are responsible for keeping their entries accurate.