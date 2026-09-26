# Paragraph wording expansion — 2026-09-26

23 annotated locations adjusted within the requested word-count ranges. Prior source files are in before/. No layout parameters or numerical results were changed. Hyphenated compounds count as one word.

## Sections/01_Introduction.tex (+4)

Before: This makes scores near zero unreliable for aggregation into subsets.

After: This makes scores near zero unreliable for aggregation into predictions of subset loss changes.

## Sections/01_Introduction.tex (+2)

Before: allowing score reweighting to be credited as an attribution advantage.

After: allowing score reweighting to be credited as an advantage in attribution quality.

## Sections/01_Introduction.tex (+2)

Before: method for Flow Matching, termed

After: method for Flow Matching image generation, termed

## Sections/01_Introduction.tex (+2)

Before: select reliable attribution scores for aggregation.

After: select reliable attribution scores for subsequent subset aggregation.

## Sections/01_Introduction.tex (+5)

Before: through the recovery of injected training samples.

After: through the recovery of injected training samples from the full training set.

## Sections/03_Preliminary.tex (+1)

Before: is the Hessian Mtrix.

After: is the training-loss Hessian matrix.

## Sections/03_Preliminary.tex (+2)

Before: The resulting score is

After: The resulting sample-wise attribution score is

## Sections/04_Method.tex (+2)

Before: different sensitivities to noise in score estimation.

After: different sensitivities to noise arising during repeated score estimation.

## Sections/04_Method.tex (+3)

Before: The relative sampling variation within each bin is

After: The relative sampling variation within each bin is then quantified as

## Sections/04_Method.tex (+1)

Before: reducing their relative weight through power transformations

After: reducing their relative aggregation weight through power transformations

## Sections/04_Method.tex (+2)

Before: Additional settings and results appear in App.

After: Additional experimental settings and detailed results appear in App.

## Sections/04_Method.tex (+1)

Before: we define the empirical SNR and retain samples exceeding a common threshold

After: we define the empirical SNR and retain samples exceeding a common detection threshold

## Sections/04_Method.tex (+3)

Before: while setting the remaining scores to zero.

After: while setting the remaining scores to zero before subset aggregation.

## Sections/04_Method.tex (+2)

Before: serve as diagnostics rather than inputs to the evaluator.

After: serve as sampling-noise diagnostics rather than required inputs to the evaluator.

## Sections/04_Method.tex (+1)

Before: the fraction of injected samples recovered among the top $K$:

After: the fraction of injected samples recovered among the top $K$ samples:

## Sections/05_Experiment.tex (+2)

Before: Full details about SNR-LDS implementation are given

After: Full details about the SNR-LDS implementation procedure are given

## Sections/05_Experiment.tex (+1)

Before: the complete threshold and reweighting protocol.

After: the complete threshold and reweighting evaluation protocol.

## Sections/05_Experiment.tex (+3)

Before: Additional settings and coverage details appear

After: Additional model settings and method-specific query coverage details appear

## Sections/05_Experiment.tex (+3)

Before: the leading method on CIFAR-2 FM generation changes with the threshold.

After: the leading method on CIFAR-2 FM generation changes as the SNR selection threshold varies.

## Sections/05_Experiment.tex (+3)

Before: These differences show that overall rankings can conceal variation across concept groups.

After: These differences show that overall rankings can conceal variation in retrieval performance across concept groups.

## Figures/Fig1/overview.tex (+2)

Before: Setup and full results are in

After: Experimental setup and full results are provided in

## Figures/Table1/table.tex (+6)

Before: Available coverage statistics are reported in

After: Available coverage statistics for individual methods and query tracks are reported in

## Figures/Table2/table.tex (+8)

Before: All 13 methods are evaluated on the same test queries within each model setting.

After: All 13 methods are evaluated on the same test queries within each model setting, with injected samples serving as known retrieval targets.

