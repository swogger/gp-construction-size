Quantifying the Cost of Construction Size for Genetic Programming in Single-Operator Grammars
Code, data and results

Dimitar Georgiev Kazakov, Technical University of Varna
Contact: dimitar.kazakov@gmail.com
Paper: [citation and DOI to be added on publication]
Repository: https://github.com/swogger/gp-construction-size
Archive: [Zenodo DOI to be added at release]
Preregistrations: OSF du2pr (first study, H1-H5) and OSF 2jqhr (held-out study, H6-H8)


CONTENTS

src/reach/        Grammars with faithful extended-real semantics, the exhaustive enumerator, tree GP
                  (legacy mode reproduces EML-GP exactly), the (1+4)-ES, target generation, the run
                  queue and the registered analyses (src/reach/analysis).
tests/            Unit and regression tests (pytest).
legacy/           Verbatim copy of the EML-GP code and stored results used for the regression test.
scripts/          Launchers for the run phases, and reproduce_results.py.
data/targets/     Every target set (random and calculator targets, held-out grammars).
data/family/      Shortest constructions of the calculator functions, locality measures.
data/enumeration/ Enumeration statistics per grammar and size (counts.json; level files are not included).
data/heldout/     The frozen predictor and its per-target predictions, as registered.
data/runs/        Manifest and results (one JSON line per run) of every run phase.
data/rescored/    Runs re-scored after the signed-zero correction of the exact-recovery check.
paper/scripts/    Every post hoc analysis, table and figure in the paper.
paper/results/values/  Every number quoted in the paper, as written by the analysis scripts.
paper/figures/    The figures.
registration/     Verification against the registered versions (see below), and the checksums
                  attached to the OSF registrations.


REPRODUCING

Python 3.9 with the pinned versions in requirements.txt:

    python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest
    .venv/bin/python scripts/reproduce_results.py

reproduce_results.py reruns every analysis from the released run data and checks that each quoted
number is reproduced. Rerunning the GP runs themselves uses the run queue
(python -m reach.runner run PHASE; phases in src/reach/runner/phases.py) and takes days of CPU.


VERIFYING AGAINST THE REGISTRATIONS

    python3 registration/verify_release.py

checks, by git tree hash, that the released code and the target, construction and held-out
prediction data are byte-identical to the versions registered on OSF. The code was changed after
the second registration only by adding options for the post hoc sensitivity analyses, with every
default unchanged; those changes are listed in registration/src_reach_since_3182bf1.patch and
are reversed by the check before hashing.


LICENCE

MIT License (see LICENSE), except legacy/: the EML-GP code is under the Apache License 2.0
(legacy/LICENSE), and legacy/eml-skill keeps its own MIT licence (legacy/eml-skill/LICENSE).
