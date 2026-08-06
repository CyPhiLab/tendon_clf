# spirob_horz: bringing up ID-CLF-QP

Notes from porting the `id_clf_qp` controller to the horizontal SpiRob model
(`mujoco_models/spirob/spirob_horz.xml`). Everything below is measured from the
compiled model, not inferred from the XML.

**Status: the port is done.** All six issues below are resolved, plus a seventh
found during bring-up (permanent inter-segment contact). The measured effect of
each fix is in [Results](#results).

## Environment

Run in the `nonlinear` conda env:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate nonlinear
```

mujoco **3.10.0**, cvxpy 1.9.1, numpy 2.2.6. The pin matters: `robot.get_mass_matrix()`
calls `mj_fullM(model, data, dst)`, which is the 3.10 signature. mujoco 3.8 uses
`mj_fullM(model, dst, qM)` and raises `TypeError`.

## Model facts

| | value |
|---|---|
| `nq` / `nv` | 48 / 48 (24 segments × 2 hinges) |
| `nu` | 3 tendons |
| `na` | 0 (dcmotor has no activation state) |
| timestep | 0.002 |
| integrator | `implicitfast` |
| base body | `segment_1__configuration_default` at `(0, 0, 0.1)` |
| ee site | `ee`, on body `attachment` — exists, no change needed |
| ee at `qpos=0` | `(-0.4804, 0, 0.1)` — arm extends along **−x**, horizontally |
| cameras | **none defined** |

## Issue 1 — actuator is a `dcmotor`, not a `motor` (resolved)

The old `spirob.xml` used plain `motor` transmissions, where actuator force equals
`ctrl` and `B = actuator_moment.T` is exactly right. `spirob_horz.xml` uses:

```xml
<dcmotor name="tendon1_act" tendon="tendon1" nominal="12 10.6 3.14159"
         gear="43.478" ctrlrange="-12 0" armature="3e-3" saturation="12 4.4 0"/>
```

Probing the compiled model gives an affine, velocity-dependent force law:

```
actuator_force = 0.8833·u − 3.3741·actuator_velocity,     |actuator_force| ≤ 12
```

Verified to **0.0 max error** over 200 random `(qpos, qvel, ctrl)` samples outside
saturation. The constants follow from `nominal="V_nom τ_nom ω_nom"`:

- `k_v = τ_nom / V_nom = 10.6 / 12 = 0.8833` (force per unit ctrl at `q̇ = 0`)
- `k_e = k_v · (V_nom / ω_nom) = 0.8833 · 3.8197 = 3.3741` (back-EMF)

`Robot._probe_actuator_constants()` measures both at init rather than hardcoding
them, so retuning `nominal="..."` cannot silently invalidate the controller model.

**Consequences for the ID constraint.** MuJoCo's forward dynamics are

```
M q̈ + qfrc_bias − qfrc_passive − qfrc_constraint
    = qfrc_actuator = moment.T @ (k_v·u − k_e·actuator_velocity)
```

so the controller's `M q̈ + h = B u` needs **both**:

1. `B = moment.T * k_v` — `update_input_matrix()` now applies `k_v`. Leaving it
   out is a 0.8833 input-gain error, and costs a factor of 6 in final task error
   (see Results).
2. `h += moment.T @ (k_e * actuator_velocity)` — the back-EMF term, folded into
   `get_passive_forces()`, which all four controllers already consume via
   `h = get_bias_forces() + get_passive_forces()`. The existing `passive_sign = -1`
   for spirob is correct and unchanged.

Saturation (`|force| ≤ 12`) is affine in `u` and could be added as a per-step bound
(`u ≥ (−12 + k_e·v)/k_v`, `u ≤ (12 + k_e·v)/k_v`). It only binds when tendon speed
exceeds ~0.41 m/s, so it is still deferred — check the logs for it before concluding
that tuning is bad.

`control_limits` is now `(-12.0, 0.0)` to match the dcmotor `ctrlrange` in volts.
It had been `(-100.0, 0.0)`, so the QP planned with ~8× the authority it has and
MuJoCo silently clipped.

## Issue 2 — `B` is near-singular in the straight configuration (resolved)

Singular values of `B` (48×3), before the `k_v` scaling:

| configuration | singular values | cond |
|---|---|---|
| straight (`qpos = 0`) | `[3.0773, 3.0771, 0.0032]` | 947 |
| small bend | `[3.0784, 3.0752, 0.0040]` | 769 |
| gravity-settled | `[3.0640, 3.0626, 0.0525]` | 58 |

Three symmetric tendons: the two differential modes bend the arm, the common mode
(all three pulling equally) produces almost no net joint torque — it just compresses
the backbone.

Both mitigations are in place:

1. `pinv_rcond = 1e-2` (a robot config field, `None` elsewhere) truncates the common
   mode out of `pinv_B`.
2. `initialize_simulation_state()` lets gravity settle the arm for `settle_steps = 1000`
   (2 s) before the controller engages, instead of starting at `qpos = 0` — which is
   exactly the worst-conditioned point, where `pinv_B` amplifies by ~312.

Note that (2) makes (1) inactive in practice: at the settled configuration the
smallest singular value (0.0525) is above the `rcond` cutoff (`1e-2 × 3.064 = 0.031`),
and switching the truncation off changes the results by **zero** to all printed
digits. It is kept as a safety net for trajectories that pass near the straight
configuration, not because it is doing work today.

## Issue 3 — the arm rests on the floor (resolved: raise the base)

With the base at `z = 0.1` and a 0.48 m arm, **all 64** sampled constant-`ctrl`
settling runs ended in floor contact (26–64 contacts each); tip `z` spanned
0.000–0.201.

`initialize_simulation_state()` now raises the base to `z = 0.55`
(`SPIROB_HORZ_BASE_HEIGHT` in `robot.py`, via `model.body_pos[base_id][2]`), rather
than editing `spirob_horz.xml`, which is generated by onshape-to-robot and would
lose the edit on regeneration. The floor geom stays in the scene for visual
reference.

0.55 is validated against the measured static-equilibrium sweep: the lowest point
of the arm sits at most **0.244 m** below the base, leaving ~0.31 m of clearance.

Rejected alternatives: deleting the floor (arm droops below `z = 0` and renders
oddly); lowering the base to the ~0.30 m minimum (no margin for dynamic overshoot).

## Issue 4 — targets are on the wrong side of the robot (resolved)

`set_target(..., 'spirob')` generated vertical-robot targets, all at **+x**, while
the horizontal arm reaches into **−x**; `circular_trajectory()` had the same problem.
Both now branch on `spirob_horz` and share one definition, `_spirob_horz_circle()`
in `utils.py`: a circle in the **y-z plane** at fixed `x`, which is the plane the
arm can actually sweep.

Measured static-equilibrium workspace (ee relative to the base, 125-point `ctrl`
grid, ramped in with 25× damping so the transient dies):

| axis | range |
|---|---|
| x | −0.475 … +0.215 |
| y | −0.196 … +0.196 |
| z | −0.238 … +0.223 |

The targets sit at `x = −0.30`, radius `0.08` about `z = base − 0.03`, well inside
that set, putting each target 0.165 m from the arm's rest pose — about 35% of the
straight-arm reach, so the set-point runs are a real excursion rather than a nudge.

The radius is set by the **tracking** experiment, not the set-point one. Tracking
error is governed by tip speed (`radius × omega`), and the arm holds up only to
roughly 0.1 m/s:

| radius | omg1 | omg3 | omg5 |
|---|---|---|---|
| 0.08 | 0.17 mm (0.2% of R) | 8.2 mm (10%) | 16.0 mm (20%) |
| 0.10 | 0.17 mm (0.2%) | 14.5 mm (15%) | 49.1 mm (49%) |
| 0.16 | 4.6 mm (2.9%) | 39.1 mm (24%) | 42.3 mm (26%) |

At 0.16 everything above `omg1` is untrackable, which would make the tracking sweep
measure the plant's speed limit instead of the controller. At 0.08 all five omegas
stay usable. Set-point accuracy barely notices the change (1.9e-04 m at radius 0.08
vs 8.4e-05 m at 0.16 — both far below anything physically meaningful).

Caveat worth knowing: **none of the 125 equilibria fully settle.** Median residual
`‖q̇‖` is 0.33 across 48 DOF even after 4 s of hold at 25× damping. The tendons'
`frictionloss="1.63"` plus the permanent inter-segment contacts (Issue 7) produce a
slow stick-slip creep, so the plant never quite stops moving. Expect a small
irreducible floor on steady-state error.

## Issue 5 — `initialize_simulation_state()` wipes the model's tuned passives (resolved)

`robot.py` blanket-assigned `model.jnt_stiffness[:] = self.stiffness` and
`model.dof_damping[:] = self.damping`, overwriting the tapered per-joint
`stiffness="0.3…"` (0.300 → 0.299 → 0.298 … along the spiral) and `damping="0.2"`
that `spirob_horz.xml` sets deliberately. That assignment is now behind
`override_passives`, which is `False` for `spirob_horz` and `True` (unchanged)
everywhere else.

## Issue 6 — no cameras (resolved)

`utils.py` did `viewer.cam.fixedcamid = robot.model.camera("ortho_side").id`, which
raised for this model in non-headless mode. An `ortho_side` camera is now defined in
`spirob_horz_control.xml`, framed for the raised base, and the viewer falls back to
the free camera instead of raising when a model has none.

## Issue 7 — the segments rest on each other, always (found during bring-up)

Raising the base clears the floor but **not** all contact. Adjacent spiral segments
touch in every configuration: `0` of 125 sampled states were contact-free, with
~30 contacts and 24 distinct body pairs even with the arm hanging in free space.
Penetrations are tiny (1e-5 … 6e-4 m) — this is the spiral resting on itself, not
a modelling error.

`qfrc_constraint` is not a small correction. Along a 3 s constant-pull run:

| term | mean ‖·‖∞ | max ‖·‖∞ |
|---|---|---|
| `qfrc_constraint` | 0.182 | 0.382 |
| `qfrc_bias` | 0.143 | 0.221 |
| `qfrc_passive` | 0.182 | 0.238 |

It is *larger than gravity/Coriolis on average*, and it also carries the tendons'
`frictionloss="1.63"`, which MuJoCo solves as a constraint. Carrying it in `h`
makes the controller's dynamics model exact:

```
max |M q̈ + h − B u|   without qfrc_constraint:  1.10e-01
                       with    qfrc_constraint:  1.40e-12
```

`include_constraint_forces` (default `False`, `True` for `spirob_horz`) subtracts
`data.qfrc_constraint` in `get_passive_forces()`. The value is one step stale — it
comes from the previous `mj_step`'s forward pass — so it acts as feedforward, not
as a term the QP can plan against. That is enough: it is worth a **10×** reduction
in final task error (see Results).

This does not contradict the earlier decision to raise the base rather than model
floor contact. Floor contact is intermittent and impulsive; inter-segment contact is
permanent and slowly varying, which is what makes a one-step-stale feedforward work.

## Results

ID-CLF-QP, set-point experiment, 5 s, mean over the four targets. Each row changes
exactly one thing from the shipped configuration.

| configuration | mean final error | vs baseline |
|---|---|---|
| **shipped** (`task_dim=3`, constraint term, `rcond=1e-2`) | **9.68e-05 m** | — |
| without `qfrc_constraint` in `h` | 9.92e-04 m | **10× worse** |
| with `B` unscaled (`k_v = 1`) | 5.82e-04 m | **6× worse** |
| without the back-EMF term in `h` | 1.05e-04 m | 8% worse |
| without `rcond` truncation | 9.68e-05 m | identical |
| `task_dim = 6` | 7.03e-05 m | 27% better |

Convergence is fast and clean: from a 0.22 m initial error to under 1 mm in
0.3–1.3 s, with the Lyapunov value `V` decreasing monotonically.

The back-EMF term looks small here only because set-point runs are slow after the
initial transient; it is worth keeping since it costs nothing and makes the model
exact. It matters much more during fast tracking.

### The `task_dim` question, answered

`task_dim = 6` on a 3-actuator arm whose `B` is effectively rank 2 was expected to
waste actuation damping angular velocity to zero. Measured, it does not hurt — it is
27% *better* in final error and more consistent across targets (6.2–7.5e-5 vs
7.3–15.7e-5), at essentially the same convergence time (0.43 s vs 0.41 s mean).

The shipped value is nevertheless **`task_dim = 3`**: the difference is 0.03 mm,
far below anything physically meaningful, and `m = 3` is the honest task
specification for this arm — `compute_target_data()` zeroes the orientation rows of
the twist anyway, so `m = 6` asks the controller to regulate an orientation nobody
specified. It also keeps the QP smaller and matches the `tendon` robot. Changing it
back is a one-line edit in `robot.py`; the numbers above are what to expect.

## Still open

- Actuator saturation (`|force| ≤ 12`) is not represented in the QP. It binds only
  above ~0.41 m/s tendon speed, which the tracking runs at high `omega` do reach.
- Gains (`Kp`, `Kd`, `e`, `reg_*`) are inherited from the vertical `spirob` and were
  not swept. They already give ~0.1 mm set-point accuracy, so there was nothing to
  chase; the tracking runs are where a sweep would pay off.
- The other controllers (`impedance`, `impedance_QP`, `clf_qp`, `uosc`) have not been
  exercised on `spirob_horz`. `run_all.py` skips the same set for it as for `spirob`.
