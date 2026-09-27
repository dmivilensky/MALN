# MALN

Numerical experiments for Section VI of *Information-Theoretic Upper Bounds for Deterministic Noise in Zeroth-Order Convex Optimization*.

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
bash run_all.sh
python3 validate_landscape.py
```

Figures are written to `figures/`; numerical results and validation reports are written to `results/`. Random seeds are fixed in the scripts. Rerunning an experiment overwrites its outputs.

## Experiments

| Section | Experiment | Script | Output |
| --- | --- | --- | --- |
| VI-A | The Hiding Mechanism | `exp_mechanism.py` | Fig. 2: `figures/mechanism.pdf` |
| VI-B | Moreau Envelope Versus Universal Convolution | `exp_dimension.py` | Fig. 3: `figures/dimension.pdf` |
| VI-C | The Smooth Strongly Convex Bracket | `exp_landscape.py` | Fig. 4: `figures/landscape.pdf` |
| VI-D | Breakdown of the Two-Point Method | `exp_breakdown.py` | Table 3: `results/breakdown.csv` |

Shared mathematical routines are in `maln/`.

## Figure 4

To run only the landscape experiment with its pinned dependencies, use a separate environment:

```bash
python3 -m pip install -r requirements-landscape.txt
python3 exp_landscape.py
python3 validate_landscape.py
```

The experiment compares theoretical bounds with empirical first-failure levels on 40 frozen quadratic objectives, using 16 and 136 fixed perturbation patterns per objective. It locates level crossings without assuming monotone loss. The markers describe this finite test family, not a uniform noise guarantee.

The script saves the plot, numerical bounds, frozen instances, certificate parameters, failure witnesses, and crossing checks. To redraw the figure from saved results:

```bash
python3 exp_landscape.py --plot-only
```

## Validation

- `validate.py`: numerical checks of the constructions, bounds, and two-point methods; writes `results/validation.json`.
- `validate_landscape.py`: checks the saved landscape data and independently tests the reconstruction and level-crossing calculations; writes `results/landscape_validation.json`.

Both scripts stop on a failed assertion. Numerical checks supplement the analytical proofs.
