# spirob_horz: bringing up ID-CLF-QP

Working notes for porting the `id_clf_qp` controller to the new horizontal SpiRob
model (`mujoco_models/spirob/spirob_horz.xml`). Everything below is measured from
the compiled model, not inferred from the XML.

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

## Issue 1 — actuator is a `dcmotor`, not a `motor`

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

`m.actuator_gainprm[:, 1]` holds `V_nom/ω_nom = 3.8197` directly. Don't hardcode —
probe at init (`ctrl = -1, q̇ = 0` gives `k_v`; `q̇ ≠ 0, ctrl = 0` gives `k_e`).

**Consequences for the ID constraint.** MuJoCo's forward dynamics are

```
M q̈ + qfrc_bias − qfrc_passive = qfrc_actuator = moment.T @ (k_v·u − k_e·actuator_velocity)
```

so the controller's `M q̈ + h = B u` needs **both**:

1. `B = moment.T * k_v` — currently `update_input_matrix()` sets `B = moment.T`, an
   0.8833 input-gain error.
2. `h += moment.T @ (k_e * actuator_velocity)` — the back-EMF term is missing
   entirely. This is **not** small: at `q̇ = 0.2` with `ctrl = -6` it flipped the sign
   of one tendon's force (`-5.30 N → +1.53 N`). Fold it into `get_passive_forces()`,
   which all four controllers already consume via
   `h = get_bias_forces() + get_passive_forces()`.
   The existing `passive_sign = -1` for spirob is correct; keep it.

Saturation (`|force| ≤ 12`) is affine in `u` and could be added as a per-step bound
(`u ≥ (−12 + k_e·v)/k_v`, `u ≤ (12 + k_e·v)/k_v`). It only binds when tendon speed
exceeds ~0.41 m/s, so it is deferred — but check the logs for it before concluding
the tuning is bad.

Also note `control_limits` for spirob is `(-100.0, 0.0)` in `robot.py`, but the
dcmotor `ctrlrange` is `(-12, 0)` volts. The QP plans with ~8× the authority it has
and MuJoCo silently clips. Must become `(-12.0, 0.0)`.

## Issue 2 — `B` is near-singular in the straight configuration

Singular values of `B` (48×3):

| configuration | singular values | cond |
|---|---|---|
| straight (`qpos = 0`) | `[3.0773, 3.0771, 0.0032]` | 947 |
| small bend | `[3.0786, 3.0743, 0.0052]` | 589 |
| gravity-settled | `[3.0660, 3.0653, 0.0485]` | 63 |

Three symmetric tendons: the two differential modes bend the arm, the common mode
(all three pulling equally) produces almost no net joint torque — it just compresses
the backbone. `initialize_simulation_state()` sets `qpos[:] = 0` for spirob, i.e. it
starts at **exactly** the worst-conditioned point, where `pinv_B` amplifies by ~312.

Mitigations to try, in order:
1. `np.linalg.pinv(B, rcond=1e-2)` to truncate the common mode (make `rcond` a robot
   config field so it can be tuned).
2. Start from a slightly bent configuration rather than `qpos = 0`, or let gravity
   settle for a few hundred steps before engaging the controller.

## Issue 3 — the arm rests on the floor (resolved: raise the base)

With the base at `z = 0.1` and a 0.48 m arm, **all 64** sampled constant-`ctrl`
settling runs ended in floor contact (26–64 contacts each); tip `z` spanned
0.000–0.201. ID-CLF-QP has no contact term in `h`, so it would fight unmodeled
constraint forces continuously.

**Decision: raise the base into free space** (~`z = 0.55`, pending the droop
measurement) so the contact-free dynamics model stays valid, matching how the
helix/tendon benchmarks work. The floor geom stays in the scene for visual
reference. Do this programmatically in `initialize_simulation_state()` via
`model.body_pos[base_id][2] = self.base_height` rather than editing
`spirob_horz.xml`, which is generated by onshape-to-robot and would lose the edit on
regeneration.

Rejected alternatives: modelling contact via `qfrc_constraint` in `h` (non-smooth,
much harder to tune); deleting the floor (arm droops below `z = 0` and renders oddly).

## Issue 4 — targets are on the wrong side of the robot

`set_target(..., 'spirob')` in `utils.py` still generates vertical-robot targets, all
at **+x**, while the horizontal arm reaches into **−x**:

| | pos1 | pos2 | pos3 | pos4 |
|---|---|---|---|---|
| x | +0.240 | +0.071 | +0.014 | +0.184 |
| z | 0.324 | 0.268 | 0.098 | 0.155 |

Unreachable, so the QP just saturates. `circular_trajectory()` has the same problem
for the `tracking` experiment. Both need a `spirob_horz` branch, with targets drawn
from the measured free-space workspace (see `scratchpad/droop.py`, which sweeps a
5×5×5 `ctrl` grid with the floor disabled and reports the reachable cloud).

## Issue 5 — `initialize_simulation_state()` wipes the model's tuned passives

`robot.py` blanket-assigns `model.jnt_stiffness[:] = self.stiffness` and
`model.dof_damping[:] = self.damping`. For spirob those are 0.01 and 0.05, which
overwrite the tapered per-joint `stiffness="0.3…"` (0.300 → 0.299 → 0.298 … along the
spiral) and `damping="0.2"` that `spirob_horz.xml` sets deliberately. The new robot
entry should skip this overwrite and keep the XML values.

## Issue 6 — no cameras

`utils.py` does `viewer.cam.fixedcamid = robot.model.camera("ortho_side").id`, which
raises for this model in non-headless mode. Add an `ortho_side` camera to
`spirob_horz_control.xml`. Headless runs are unaffected.

## Plan

1. New robot key `spirob_horz` in `robot.py` / `run.py` (keeps the old vertical
   `spirob` config, targets, and `results/` data intact for comparison).
   `spirob_control.xml` has been restored to `include spirob.xml`; the horizontal
   scene now lives in `spirob_horz_control.xml`. `_load_model` needs a path override
   so `spirob_horz` resolves to `mujoco_models/spirob/spirob_horz_control.xml` —
   the scene must stay in the `spirob/` directory because `meshdir="assets"` resolves
   relative to it.
2. Probe-based dcmotor constants; fix `B` scaling and the back-EMF term in `h`.
3. `control_limits = (-12.0, 0.0)`; `pinv` with `rcond`.
4. Raise the base; re-measure the workspace; pick 4 targets and a trajectory.
5. Keep XML stiffness/damping.
6. Add the `ortho_side` camera.
7. Tune `Kp`, `Kd`, `e`, `reg_qdd`, `reg_u`, `reg_null`, `reg_dl` against
   `task_error` and the Lyapunov trace `V`.

## Open question

`task_dim` is 6 for spirob, but `compute_target_data()` always sets the orientation
rows of `twist` to zero — so the controller spends actuation damping angular velocity
to zero on a 3-actuator arm whose `B` is effectively rank 2. Worth testing
`task_dim = 3` for `spirob_horz`; `compute_target_data()` already handles `m == 3`.
