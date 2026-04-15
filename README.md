# Chemical Equation Balancer

A Python/tkinter application demonstrating two linear algebra techniques — **Null Space** and **Least Squares** — applied to chemical equation balancing and stoichiometric ratio recovery from noisy sensor data.

Developed for **UE24MA241B — Linear Algebra and its Applications**.

---

## The Mathematics

| | Problem | Technique | Solver |
|---|---|---|---|
| **Act 1** | Find stoichiometric coefficients | Null space of `Ax = 0` | `scipy.linalg.null_space` |
| **Act 2** | Recover ratios from noisy readings | Least squares on `Ax ≈ b` | `numpy.linalg.lstsq` |

### Act 1 — Null Space

Conservation of mass requires that atom counts balance across both sides of a reaction. This is encoded as a homogeneous linear system `Ax = 0`, where:

- **A**: atom-count matrix (rows = elements, columns = molecules; reactant columns positive, product columns negative)
- **x**: unknown stoichiometric coefficients

The null space of **A** contains all vectors **x** satisfying this constraint. The first basis vector is scaled to the smallest positive integers to yield the balanced equation.

**Example — Methane combustion:**

```
         CH4   O2   CO2   H2O
   C  [  1     0   -1     0  ]
   H  [  4     0    0    -2  ]
   O  [  0     2   -2    -1  ]
```

Null space → `[1, 2, 1, 2]` → **CH₄ + 2O₂ → CO₂ + 2H₂O** ✓

### Act 2 — Least Squares

Real sensors introduce noise, producing an overdetermined system (`Ax ≈ b`) with no exact solution. The app minimises the squared residual `‖Ax − b‖²` via the normal equations:

$$x = (A^T A)^{-1} A^T b$$

Recovered ratios are compared against the Act 1 ground truth to demonstrate reconstruction accuracy under noise.

---

## Features

- **6 built-in reactions** — NH₃ combustion, H₂ combustion, CH₄ combustion, photosynthesis, propane combustion, H₂SO₄ acid-base
- **Custom equation builder** — add/remove reactants, products, and elements; edit the atom-count matrix directly
- **Signed matrix display** — see exactly what is passed to the solver
- **Atom-by-atom verification table** — confirms conservation for every element
- **Interactive sensor grid** — view and edit noisy readings before solving
- **Bar chart comparison** — recovered vs. true ratios for each molecule
- **Residual readout** — quantifies noise and recovery quality
- **Adjustable parameters** — noise level (0–50%) and number of readings (5, 8, 12, or 20)

---

## Getting Started

**Prerequisites:** Python 3.9+, pip

```bash
pip install numpy scipy
```

> On some Linux distributions, tkinter must be installed separately:
> ```bash
> sudo apt-get install python3-tk
> ```

**Run (script):**
```bash
python chemical_balancer.py
```

**Run (notebook):**

Open `chemical_balancer.ipynb` in Jupyter and run all cells. Each section of the code is in its own cell. Launch the app by running the final cell.

---

## How to Use

**Act 1 — Balance an equation**

1. Select a preset reaction, or click `+ Reactant` / `+ Product` and `+ Add Element` to build your own.
2. Fill in the atom count matrix (positive integers only — signs are applied automatically).
3. Click **✓ Solve — Find Null Space**.
4. The balanced equation and verification table appear below.

**Act 2 — Sensor recovery**

1. After solving Act 1, click **Open Act 2 →**.
2. Adjust noise level and reading count, then click **Generate Readings**.
3. Click **✓ Solve — Least Squares**.
4. Bar charts show recovered vs. true ratios; the residual norm indicates recovery quality.

---

## Project Structure

```
chemical-balancer/
├── chemical_balancer.ipynb   # Notebook: one cell per logical section
├── chemical_balancer.py      # Equivalent single-file script
└── README.md
```

**Notebook cell layout:**

| Cell | Contents |
|---|---|
| Imports | Standard library and third-party imports |
| Theme & Constants | Colour palette and layout padding |
| Examples | 6 predefined reactions |
| Math Engine | `_to_integers`, `build_signed_matrix`, `solve_null_space`, `solve_least_squares`, `generate_sensor_readings` |
| UI Helpers | `styled_btn`, `make_card`, `scrollable_frame` |
| Act 2 Window | `Act2Window` class |
| Act 1 Main Window | `ChemicalBalancerApp` class |
| Launch | `app.mainloop()` |

---

## Technical Stack

| Library | Usage |
|---|---|
| **NumPy** | Matrix operations, `linalg.lstsq` for least squares |
| **SciPy** | `linalg.null_space` for null space computation |
| **Tkinter** | Cross-platform GUI |

---

## Team Members

| Name | USN |
|---|---|
| Aarav Yuval | PES1UG24AM004 |
| A Ravi Teja | PES1UG24AM001 |
| Ahan A Mysore | PES1UG24AM019 |
