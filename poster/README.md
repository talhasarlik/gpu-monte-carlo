# Poster — GPU-Accelerated Monte Carlo

Source for the end-of-semester academic poster. Self-contained: every
number on the poster traces back to a file in this repo.

## Layout

```
poster/
├── poster.tex              main LaTeX source (a0poster, 3 columns)
├── content_sheet.md        plain-text mirror of every poster string
├── figures/
│   ├── make_figures.py     regenerates all 5 PDFs from results/benchmark.csv
│   ├── fig_throughput.pdf
│   ├── fig_speedup.pdf
│   ├── fig_speedup_bars.pdf
│   ├── fig_convergence.pdf
│   └── fig_nsight_metrics.pdf
└── diagrams/
    ├── algorithm_flow.tex          \input{} into poster.tex
    ├── reduction_tree.tex          \input{} into poster.tex
    └── kernel_architecture.tex     used by make_figures.py? no — \input{} too
```

The TikZ diagrams in `diagrams/` are `\input{}`-ed directly by
`poster.tex`. The `figures/` PDFs are produced by matplotlib from the
benchmark CSV plus hard-coded Nsight numbers transcribed from
`results/mc_*_summary.txt`.

## Compile

Requires a TeX distribution with `a0poster`, `tcolorbox`, `tikz`,
`booktabs`, `microtype`, `lmodern` (all included in MiKTeX-full,
TeX Live 2023+, MacTeX).

```powershell
cd poster
pdflatex poster.tex
pdflatex poster.tex          # run twice so cross-references settle
```

Output: `poster.pdf` (A0 portrait, ~841 × 1189 mm).

If you change a benchmark or a Nsight metric:

```powershell
cd poster\figures
py make_figures.py
```

This re-reads `..\..\results\benchmark.csv`, re-emits every PDF, and
prints the headline numbers so you can sanity-check them against the
poster prose.

## Edit prose without LaTeX

`content_sheet.md` contains every string from the poster in section
order, in plain Markdown. Edit there first, then copy the changed
block into the matching `\begin{posterblock}{...}` in `poster.tex`.

## Numbers — provenance

| What                        | Source                                     |
|-----------------------------|--------------------------------------------|
| Speedup figures             | `results/benchmark.csv` (median of 5 runs) |
| Validation table @ N=10⁶    | `docs/plan.md` lines 60-67                 |
| Nsight metrics              | `results/mc_*_summary.txt` (Section: GPU Speed Of Light) |
| Architectural narrative     | `docs/plan.md` log entries from 2026-05-18 |
| Math / Black-Scholes        | `src/common/black_scholes.h` + Glasserman ch. 3 |

## Customisation cheat sheet

- **Colour palette** — top of `poster.tex`:
  `\definecolor{ituBlue}{RGB}{0,60,120}` etc. Change these once.
- **Title size** — `\Huge\bfseries ... \par` in the title banner.
- **Column widths** — three `\begin{minipage}[t]{0.315\textwidth}`
  blocks; bump to `0.32` and trim margins if columns wrap.
- **Block style** — single `posterblock` tcolorbox definition; change
  arc radius, padding, or border colour once for all blocks.
- **Add a section** — wrap content in
  `\begin{posterblock}{Heading}...\end{posterblock}` inside whichever
  minipage column should hold it.

## Known gotchas

- `pdflatex` must be run twice for cross-references and tcolorbox
  geometry to settle.
- The Turkish character `ş` (Sarlık) needs `\usepackage[utf8]{inputenc}`
  + `\usepackage[T1]{fontenc}` (already in the preamble).
- `a0poster` ignores `\fontsize` outside of its own scaled scheme; use
  `\Huge`, `\LARGE`, `\large` etc. rather than absolute point sizes.
- If `tikzposter` is preferred over `a0poster`, swap the document class
  and replace each `posterblock` with `\block{Heading}{...}`. The
  content (`content_sheet.md`) is structured to make that swap easy.
