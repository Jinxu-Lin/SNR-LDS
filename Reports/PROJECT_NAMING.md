# SNR-LDS project naming

The current project and repository name is **SNR-LDS**. The installable Python
distribution is **snr-lds**, version 0.2.0. Current source metadata, public
documentation, report entry points, and newly generated scoring provenance use
these names.

## Deliberately retained names

- **FMAS** is the Flow Matching Attribution Score method, not a project alias.
- **balds**, **balds-run**, **balds-repeat**, the balds import package, and
  BALDS_* environment variables remain compatible with existing scripts.
- **CFA-PRED-PILOT-v1** and the cfa.ndarray.v1 hash domain are scientific
  identity constants. Renaming them would alter reproducibility identities.
- Historical source paths in provenance manifests, migration inventories,
  dated receipts, and CodeSnapshots are not rewritten. Their old names
  describe the original inputs or software versions, not the current release.
- The local checkout directory may still be named BA-LDS; GitHub and package
  branding do not require moving the user's workspace.

New default score-producer metadata uses snr-lds-0.2.0. Explicit code_version
overrides remain supported; existing artifacts and their recorded identities
are not rewritten. Do not silently mix newly produced blocks with archived
blocks carrying a different code version.

## Installation compatibility

Install from Codes/ with python -m pip install . as before. The distribution
name change does not move the import package or CLI entry points. Use a clean
virtual environment when moving from the old ba-lds distribution, since both
distributions own the same balds package and command files.

The base install supports the CPU evaluator. The balds-run training/workflow
entry point requires the train optional dependencies, including PyTorch.

## Validation (2026-09-26)

- Clean virtual-environment installation produced snr-lds 0.2.0 and retained
  the three existing console entry points.
- balds and balds-repeat help passed with base dependencies; balds-run help
  passed in the environment containing the train dependencies. All display
  SNR-LDS branding.
- Full CPU regression suite: **329 passed** in 91.68 seconds, including
  distribution naming, compatibility identifiers, and scoring provenance tests.
- The remaining old-name strings in active source are the two explicitly
  preserved scientific identity domains, not project descriptions.
