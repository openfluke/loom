# Lucy Bloom Rivers

Architecture shorthand for a Loom stack that combines **multi-region volumetric** layout, **bicameral** train vs run, **discrete-time stepping** (`step.go`), and **streaming** inference on the outside — summarized as **MRBiVS** (**M**ulti-**R**egion · **Bi**cameral · **V**olumetric · **S**tep).

---

## Letter expansion

| Word | Letters | Meaning |
|------|---------|---------|
| **Lucy** | — | Spoken handle only (no MRBiVS letters required here). |
| **Bloom** | **B**, **M** | **Bi**cameral · **M**ulti-region mesh |
| **Rivers** | **R**, **i**, **V**, **S** | **R**outing / regional links · **i** completes **Bi** (with **B** from Bloom) · **V**olumetric grid · **S**tep mesh + streaming |

### Initialisms

**L.U.C.Y.** — *Lattice Unified Clock Yoked-net.*

**B.L.O.O.M.** — *Bicameral Loom Open-grid Orchestration Multi-region.*

**R.I.V.E.R.S.** — *Routed In Volumetric Engines Rhythmically Stepping.*

---

## Architecture

- **Volumetric network** — Grid of layers (`VolumetricNetwork`), not just depth stacked one way. Multi-region layouts: branches, combine modes, optional remote regional links (e.g. `glitch/measure/regional_mix`).

- **Bicameral** — Train vs run hemispheres with periodic mirror/sync (e.g. `glitch/systolic_demo_bicameo`).

- **Step mesh** — Inner state advances in ticks: `StepState`, `StepForward` in [`poly/step.go`](../poly/step.go); see [`docs/step.md`](../docs/step.md).

- **Streaming decode** — Outer loop can stay standard autoregressive / KV-style; mesh stepping is the inner temporal loop.

- **KV cache** — Ordinary attention cache where used; align with mesh ticks per design.

---

## Test output

Full layer-matrix runs (parity tables, training matrices, save/reload rows) are often captured under:

- `lucy/lucy_testing_output/log.txt`

How to read the summary symbols and what the rows imply (including “H-DRIFT”, Save/Reload FAIL vs TrainOK, and the peak gap footer) is documented in [`docs/testing_and_validation.md`](../docs/testing_and_validation.md).
