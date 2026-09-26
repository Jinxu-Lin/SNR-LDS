# Appendix layout review

## Before editing

Inspected rendered PDF pages 13–33, saved as render/before-*.png. Page numbers below refer to this baseline, not the final pagination.

| Pages | Location | Observation and intended adjustment |
| --- | --- | --- |
| 13 | A opening; A.2 introduction and loss convention | Short final lines; tighten or modestly expand existing explanations. |
| 13 | A.1 stationarity, zero-perturbation, deletion endpoint | Three isolated one-line equation introductions; explain the differentiation, local expansion, and rescaling steps. |
| 14 | A.2 curvature introduction | One-line formula lead-in; explain the GGN construction without changing assumptions. |
| 14 | A.3 pairwise-gap introduction | Single-line lead-in; explain which deletions contribute to the difference. |
| 14–15 | A.3 covariance/rank reversal; A.4 assumptions and closing sentence | Short tails, including “diction,” “vanishes,” and “ment”; tighten wording. |
| 15–16 | B.1 diffusion/FM and evaluation paragraphs; B.2 gradient features | Short tails; trim redundant phrases or clarify existing distinctions. |
| 16–17 | B.2 output readouts, projected kernel, final-checkpoint sentence | Formula introductions or isolated sentence too short; move checkpoint information into the preceding paragraph and expand mathematical transitions. |
| 18–19 | C.1 conventions; C.3/C.4/C.5 opening sentences | Several left-ending tails; add scope information already established by these experiments. |
| 21 | C.5 matched-DAS paragraph | Single-word “features” tail; shorten the final sentence. |
| 22–23 | D.1 scale formula and reporting; D.2 introduction and ending | Short tails; clarify the fitted coefficient and tighten protocol prose. |
| 24–26 | E.1 opening; E.2 opening; E.3 control results; E.4 coverage | Short tails; adjust local wording without changing statistics or claims. |
| 29–31 | Figure/table captions | Review short caption tails after text reflow; keep numerical results and figure colors unchanged. |
| 32–33 | F notation | Keep tabular entries concise; remove revision/link colors, but do not pad cells for alignment. |

## Constraints

Do not modify main-text source files, numerical results, assumptions, or equation definitions. Disable appendix revision and hyperlink colors locally; preserve plot encodings. Avoid forced line breaks, font changes, or artificial spacing to fill prose lines. Recheck rendered output after edits.

## Changes and verification

- Disabled revision coloring and colored hyperlinks at the appendix boundary. Plot colors and hyperlink functionality remain intact.
- Rephrased short paragraph endings across A–F and several appendix captions; used repeated compile/render checks rather than estimating line length from source text alone.
- Expanded the A.1 stationarity, parameter-response, and deletion-rescaling transitions; explained GGN curvature in A.2, the pairwise prediction gap in A.3, and cross terms in A.4. Expanded short formula introductions in B.2 and D.1; moved the isolated final-checkpoint sentence into its preceding paragraph.
- Preserved mathematical formulas, numerical experimental results, and evaluation assumptions. Kept notation-table cells concise instead of padding them.
- Compared extracted text for pages 1–12 against the saved baseline: identical.
- LaTeX compilation passed without undefined references or overfull-box warnings. The paper remains 33 pages.
- Before/after rendered pages are saved under render/. Original appendix sources and affected captions are preserved under before/.
- Final visual checks focus on prose and caption tails; headings, table cells, plot labels, and unavoidable page-boundary continuations are not treated as paragraph-layout defects.
