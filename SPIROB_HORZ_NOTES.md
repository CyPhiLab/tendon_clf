# spirob_horz: bringing up ID-CLF-QP

Notes from porting the `id_clf_qp` controller to the horizontal SpiRob. Everything
below is measured from the compiled model, not inferred from the XML.

**Status: the port is done.** All six issues below are resolved, plus a seventh
found during bring-up (permanent inter-segment contact). Per-fix measurements are
in [Results](#results).

## Where the model comes from

The robot is maintained in [CyPhiLab/spirob_mujoco](https://github.com/CyPhiLab/spirob_mujoco),
pinned here as a submodule at `external/spirob_mujoco`. This repo owns only the
scene around it — raised base, mocap target, `ortho_side` camera, floor, keyframes —
in `mujoco_models/spirob/spirob_horz_control.xml`.

```bash
git clone --recurse-submodules ...        # or, in an existing clone:
git submodule update --init
```

**The `<compiler meshdir>` line must come after the `<include>`, not before.**
MuJoCo resolves meshes as `main_dir / meshdir / include_dir / file`, taking
`meshdir` from the *included* file. Declared before the include, the model's own
`meshdir="assets"` wins and then resolves against the scene's directory, which
fails with a path like `assets/spirob/segment_17.stl`. Declared after, the scene's
value wins. Upstream's own `models/scenes/free.xml` uses the same pattern.

Measured, not guessed: a `<compiler>` before the include fails, after it loads.
Neither `MjSpec.attach()` nor symlinks are needed, though both also work — note
that `attach` resolves option conflicts *in favour of the parent* and only warns,
so a scene that attaches rather than includes must repeat the model's `<option>`
block verbatim, `<flag filterparent="disable"/>` included. Without that flag
adjacent segments stop colliding entirely.

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
| timestep | 0.001 |
| integrator | `implicitfast`, elliptic cones, `filterparent` disabled |
| base body | `segment_1__configuration_default` at `(0, 0, 0.1)` |
| ee site | `ee`, on body `attachment` — exists, no change needed |
| ee at `qpos=0` | `(-0.4804, 0, 0.1)` — arm extends along **−x**, horizontally |
| joint stiffness / damping | 0.30 uniform / 0.10 |
| contacts at `qpos=0` | 12 |
| cameras | none in the model; `ortho_side` is added by the scene |

## Issue 1 — actuator is a `dcmotor`, not a `motor` (resolved)

The old `spirob.xml` used plain `motor` transmissions, where actuator force equals
`ctrl` and `B = actuator_moment.T` is exactly right. The horizontal model uses:

```xml
<dcmotor name="tendon1_act" tendon="tendon1" nominal="12 10.6 3.14159"
         gear="43.478" ctrlrange="-12 0" armature="3e-3" saturation="12 4.4 0"/>
```

Probing the compiled model gives an affine, velocity-dependent force law:

```
actuator_force = 0.8833·u − 3.3741·actuator_velocity,     |actuator_force| ≤ 12
```

The constants follow from `nominal="V_nom τ_nom ω_nom"`:

- `k_v = τ_nom / V_nom = 10.6 / 12 = 0.8833` (force per unit ctrl at `q̇ = 0`)
- `k_e = k_v · (V_nom / ω_nom) = 0.8833 · 3.8197 = 3.3741` (back-EMF)

`Robot._probe_actuator_constants()` measures both at init rather than hardcoding
them, so retuning `nominal="..."` upstream cannot silently invalidate the
controller model.

**Consequences for the ID constraint.** MuJoCo's forward dynamics are

```
M q̈ + qfrc_bias − qfrc_passive − qfrc_constraint
    = qfrc_actuator = moment.T @ (k_v·u − k_e·actuator_velocity)
```

so the controller's `M q̈ + h = B u` needs both `B = moment.T * k_v` and the
back-EMF term `h += moment.T @ (k_e * actuator_velocity)`, the latter folded into
`get_passive_forces()`, which all four controllers already consume. The existing
`passive_sign = -1` for spirob is correct and unchanged.

With every term present the model is exact — `max |M q̈ + h − B u| = 9.8e-15` over
60 random states. Dropping individual terms, the open-loop residual is:

| term omitted | residual |
|---|---|
| back-EMF | 4.48 |
| `k_v` scaling on `B` | 1.11 |
| `qfrc_constraint` | 0.079 |

Saturation (`|force| ≤ 12`) is affine in `u` and could be added as a per-step bound
(`u ≥ (−12 + k_e·v)/k_v`, `u ≤ (12 + k_e·v)/k_v`). It only binds above ~0.41 m/s
tendon speed, so it is deferred — but the fast sweep demo does reach that.

`control_limits` is `(-12.0, 0.0)` to match the dcmotor `ctrlrange` in volts. It had
been `(-100.0, 0.0)`, so the QP planned with ~8× the authority it has and MuJoCo
silently clipped.

## Issue 2 — `B` is near-singular in the straight configuration (resolved)

Singular values of `B` (48×3), before the `k_v` scaling:

| configuration | singular values | cond |
|---|---|---|
| straight (`qpos = 0`) | `[3.0773, 3.0771, 0.0032]` | 947 |
| gravity-settled | `[2.6996, 2.6915, 0.0574]` | 47 |

Three symmetric tendons: the two differential modes bend the arm, the common mode
(all three pulling equally) produces almost no net joint torque — it just compresses
the backbone.

Both mitigations are in place:

1. `pinv_rcond = 1e-2` (a robot config field, `None` elsewhere) truncates the common
   mode out of `pinv_B`.
2. `initialize_simulation_state()` lets gravity settle the arm for `settle_time = 2.0`
   seconds before the controller engages, instead of starting at `qpos = 0` — exactly
   the worst-conditioned point, where `pinv_B` amplifies by ~312. (Specified in
   seconds, not steps, so it survives an upstream timestep change.)

Note that (2) makes (1) inactive: at the settled configuration the smallest singular
value (0.0574) is above the `rcond` cutoff (`1e-2 × 2.6996 = 0.027`), and switching
the truncation off changes the results by **zero** to all printed digits. It is kept
as a safety net for trajectories passing near the straight configuration, not
because it is doing work today.

## Issue 3 — the arm rests on the floor (resolved: raise the base)

With the base at `z = 0.1` and a 0.48 m arm, every sampled constant-`ctrl` settling
run ended in floor contact. `initialize_simulation_state()` now raises the base to
`z = 0.55` (`SPIROB_HORZ_BASE_HEIGHT` in `robot.py`, via `model.body_pos[base_id][2]`)
rather than editing the model, which lives in the submodule. The floor geom stays in
the scene for visual reference.

Clearance is ample: settled, the lowest point of the arm sits at `z = 0.441`, i.e.
0.109 m of droop against 0.55 m of height.

## Issue 4 — targets are on the wrong side of the robot (resolved)

`set_target(..., 'spirob')` generated vertical-robot targets, all at **+x**, while
the horizontal arm reaches into **−x**; `circular_trajectory()` had the same problem.
Both now branch on `spirob_horz` and share one definition, `_spirob_horz_circle()`
in `utils.py`: a circle in the **y-z plane** at fixed `x`, the plane the arm sweeps.

Targets sit at `x = −0.30`, radius `0.08` about `z = base − 0.03`, putting each one
~0.15 m from the arm's rest pose. The radius is set by the **tracking** experiment,
not the set-point one: tracking error is governed by tip speed (`radius × omega`) and
the arm holds up only to roughly 0.1 m/s, so a radius of 0.16 leaves everything above
`omg1` untrackable and would measure the plant's speed limit instead of the
controller.

There is also a horizontal sweep demo, `--experiment tracking --omega sweep`, with
the rate set by `--sweep-omega` (rad/s) or `--sweep-hz`. It sweeps in the **x-y**
plane — perpendicular to gravity — along an arc at the radius of the arm's own
length. That arc is deliberately outside the reachable set, so a fixed share of its
tracking error is a radial shortfall; the demo is about the shape of the motion.

## Issue 5 — `initialize_simulation_state()` wipes the model's tuned passives (resolved)

`robot.py` blanket-assigned `model.jnt_stiffness[:] = self.stiffness` and
`model.dof_damping[:] = self.damping`, overwriting what the model sets deliberately
(0.30 stiffness, 0.10 damping) with the vertical robot's 0.01 and 0.05. That
assignment is now behind `override_passives`, `False` for `spirob_horz` and `True`
(unchanged) everywhere else.

## Issue 6 — no cameras (resolved)

`utils.py` did `viewer.cam.fixedcamid = robot.model.camera("ortho_side").id`, which
raised for this model in non-headless mode. The scene now defines an `ortho_side`
camera framed for the raised base, and the viewer falls back to the free camera
instead of raising when a model has none.

## Issue 7 — the segments rest on each other (found during bring-up)

Raising the base clears the floor but not all contact: adjacent spiral segments touch
in every configuration, 9–13 contacts even with the arm hanging in free space.
Penetrations are tiny — this is the spiral resting on itself, not a modelling error.

`qfrc_constraint` is not negligible. Along a 3 s constant-pull run:

| term | mean ‖·‖∞ | max ‖·‖∞ |
|---|---|---|
| `qfrc_constraint` | 0.158 | 0.192 |
| `qfrc_bias` | 0.140 | 0.211 |
| `qfrc_passive` | 0.181 | 0.259 |

It still exceeds gravity/Coriolis on average, and it also carries the tendons'
`frictionloss="1.63"`, which MuJoCo solves as a constraint — that is likely most of
it now. `include_constraint_forces` (default `False`, `True` for `spirob_horz`)
subtracts `data.qfrc_constraint` in `get_passive_forces()`. The value is one step
stale, so it acts as feedforward rather than something the QP plans against.

This does not contradict raising the base rather than modelling floor contact. Floor
contact is intermittent and impulsive; inter-segment contact is permanent and slowly
varying, which is what makes a one-step-stale feedforward work.

**This is the finding that surfaced the upstream model bug.** Against the older,
diverging copy of the model the arm self-contacted in *every* sampled state, 30–44
contacts at a time, and carrying `qfrc_constraint` was worth 8× in final task error.
Upstream's fix cut that to 9–13 contacts and the benefit to 2.5× — still real, but
much of the original effect was the bug, not the physics.

## Results

ID-CLF-QP, set-point experiment, 5 s, mean over the four targets, against the
submodule model. Each row changes exactly one thing from the shipped configuration.

| configuration | mean final error | time to 1 mm | vs baseline |
|---|---|---|---|
| **shipped** (`task_dim=3`, constraint term, `rcond=1e-2`) | 1.44e-04 m | **0.62 s** | — |
| with `B` unscaled (`k_v = 1`) | 9.48e-04 m | 0.64 s | 6.6× worse |
| without `qfrc_constraint` in `h` | 3.64e-04 m | 0.66 s | 2.5× worse |
| without `rcond` truncation | 1.44e-04 m | 0.62 s | identical |
| `task_dim = 6` | 7.50e-05 m | 1.58 s | finer, much slower |
| without the back-EMF term in `h` | 7.29e-05 m | 0.93 s | finer, slower |

Convergence is clean: from ~0.15 m initial error to under 1 mm in 0.25–1.0 s, with
the Lyapunov value `V` decreasing monotonically. All four targets are reachable.

### Two configurations settle finer than the shipped one

Both `task_dim = 6` and dropping the back-EMF term roughly halve the final error
while taking 1.5–2.5× longer to reach a millimetre. Neither is a free win, and the
steady-state gap is 0.07 mm — far below anything physically meaningful — so the
shipped configuration optimises for convergence speed instead.

The back-EMF result is worth understanding rather than just recording, because it
looks paradoxical: the term is the *largest* source of open-loop model error when
omitted (residual 4.48, above), yet omitting it gives a lower steady-state error.
The likely mechanism is that compensating back-EMF cancels the motor's own damping,
leaving a faster but jitterier closed loop; leaving it uncompensated puts that
damping back. It is kept because it makes the model exact and converges faster, but
this is the first knob to try if steady-state jitter ever matters more than speed.

`task_dim = 6` shows the same speed/precision trade, and its cost is concentrated on
the two lateral targets (2.84 s and 2.59 s to a millimetre, against 0.88 s and 0.97 s
for `m = 3`) — those are the moves that need the differential tendon modes most, so
paying actuation to regulate an orientation that `compute_target_data()` zeroes out
hurts precisely there. `task_dim = 3` also keeps the QP smaller and matches the
`tendon` robot.

## Still open

- Actuator saturation (`|force| ≤ 12`) is not represented in the QP. It binds above
  ~0.41 m/s tendon speed, which the faster sweep rates do reach.
- Gains (`Kp`, `Kd`, `e`, `reg_*`) are inherited from the vertical `spirob` and were
  never swept. They give ~0.15 mm set-point accuracy, so there was nothing to chase;
  the tracking runs are where a sweep would pay off — especially now that upstream
  halved the damping.
- The other controllers (`impedance`, `impedance_QP`, `clf_qp`, `uosc`) have not been
  exercised on `spirob_horz`. `run_all.py` skips the same set for it as for `spirob`.
