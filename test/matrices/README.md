# Test matrices

Seven 199×199 sparse Julia linear systems collected from a real ill-conditioned
nonlinear-solve regression. Each file contains two Julia expressions:

- line 1: a `sparse(...)` call constructing the matrix `A`,
- line 2: a vector literal for the right-hand side `b`.

| file | rank | notes |
|------|------|-------|
| `linsolve_0.txt` | 198 | rank-deficient |
| `linsolve_1.txt` | 198 | rank-deficient |
| `linsolve_2.txt` | 198 | rank-deficient |
| `linsolve_3.txt` | 198 | rank-deficient |
| `linsolve_4.txt` | 199 | full rank |
| `linsolve_5.txt` | 199 | full rank |
| `linsolve_6.txt` | —   | pathological: `b` contains `NaN` |

Used by the test suite to verify that `scpqr` produces:

- residuals matching SPQR / SVD-pseudoinverse on the rank-deficient cases,
- finite `x` on every case where `b` is finite,
- and proper handling of `NaN` inputs (file 6).

These are checked in as test fixtures so the suite runs unchanged on CI.
