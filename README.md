# MOSFET Design Explorer

A physics-based NMOS device simulation and design tool built from first principles, following Streetman & Banerjee *Solid State Electronic Devices*.

> **Status:** Week 1 of 8 — project scaffolded.

---

## What this project does

The MOSFET Design Explorer lets you interactively explore how device parameters — oxide thickness, substrate doping, channel dimensions, temperature, and bias voltages — affect MOSFET behavior. It computes threshold voltage, I-V characteristics, and small-signal parameters directly from semiconductor physics rather than using look-up tables or SPICE models.

The goal is a deployed web application where a user can ask: *"If I change tox from 5 nm to 3 nm and double the substrate doping, what happens to VT, ID,sat, and subthreshold swing?"* — and see the answer immediately.

---

## Physics scope

See [`docs/physics_model.md`](docs/physics_model.md) for the complete equation set and assumptions.

**Version 1 includes:**
- Full charge-neutrality carrier statistics with temperature dependence
- Masetti empirical mobility model (doping-dependent µn, µp)
- MOS capacitor electrostatics: Cox, φF, VT, body effect, C-V curve
- Long-channel NMOS I-V: cutoff, triode, saturation (square-law GCA)
- Subthreshold conduction and subthreshold swing vs temperature
- Channel-length modulation (λ)
- Small-signal parameters: gm, gds, ro

**Explicitly out of scope (Version 1):** short-channel effects, velocity saturation, poly depletion, gate leakage, non-uniform doping profiles.

---

## Repository structure

```
mosfet-design-explorer/
├── src/mosfet_explorer/
│   ├── constants.py        # Physical constants, Si material params, ni(T)
│   ├── carriers.py         # n, p, EF, mobility, conductivity
│   ├── mos_capacitor.py    # Cox, φF, VT, C-V, body effect  [Week 4+]
│   ├── iv_model.py         # NMOS I-V characteristics       [Week 5+]
│   ├── small_signal.py     # gm, gds, ro                    [Week 8+]
│   └── plotting.py         # Shared matplotlib style
├── notebooks/
│   ├── exploration/        # Weekly scratch notebooks
│   └── demos/              # Polished portfolio demos
├── tests/                  # pytest unit tests
├── docs/
│   └── physics_model.md    # Full equation reference
├── assets/                 # Exported figures
├── app.py                  # Streamlit app entry point  [Week 6+]
└── requirements.txt
```

---

## Getting started

```bash
git clone https://github.com/ssddss-ctrl/mosfet-design-explorer.git
cd mosfet-design-explorer
pip install -r requirements.txt
```

---

## Built with

- Python 3.11
- NumPy, SciPy, Matplotlib
- Streamlit (Week 6+)
- pytest

---

## References

- B. Streetman & S. Banerjee, *Solid State Electronic Devices*, 7th ed.
- S. Sze & K. Ng, *Physics of Semiconductor Devices*, 3rd ed.
