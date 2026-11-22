"""Regularized capstan sliding on site-only tendons ordered from motor to tip."""

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray


class TendonFriction:
    """Distribute actuator tension over free spans; no slack or static-friction state."""

    def __init__(self, model: mujoco.MjModel, mu: ArrayLike = 0.0,
                 velocity_eps: float = 1e-4):
        self.model = model
        coefficients = np.asarray(mu, dtype=float)
        if coefficients.ndim == 0:
            coefficients = np.full(model.nu, float(coefficients))
        if (coefficients.shape != (model.nu,) or not np.isfinite(coefficients).all()
                or np.any(coefficients < 0)):
            raise ValueError('tendon_friction_mu must be nonnegative and finite: scalar or one per actuator')
        self.mu = coefficients.copy()
        self.velocity_eps = float(velocity_eps)
        if not np.isfinite(self.velocity_eps) or self.velocity_eps <= 0:
            raise ValueError('tendon_friction_velocity_eps must be finite and positive (m/s)')
        self.enabled = bool(np.any(self.mu > 0))
        self.routes: list[NDArray[np.int32]] = []
        if not self.enabled:
            return
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError('Tendon sliding friction does not support RK4; use Euler or implicitfast')
        if (np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_TENDON)
                or np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)
                or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE)
                or np.any(model.actuator_gear[:, 0] <= 0)
                or np.any(model.actuator_gainprm[:, 0] <= 0)):
            raise ValueError('Tendon sliding friction requires positive-gear, stateless tendon motors')
        for tendon in model.actuator_trnid[:, 0]:
            start, count = model.tendon_adr[tendon], model.tendon_num[tendon]
            if (count < 2 or np.any(
                    model.wrap_type[start:start + count] != mujoco.mjtWrap.mjWRAP_SITE)):
                raise ValueError('Tendon sliding friction requires site-only paths, ordered motor to tip')
            self.routes.append(model.wrap_objid[start:start + count].copy())

    def span_kinematics(self, data: mujoco.MjData, actuator: int
                        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return span-length Jacobians, interior turning angles, and guide slip velocities."""
        sites = self.routes[actuator]
        edges = np.diff(data.site_xpos[sites], axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
            raise ValueError(f'Degenerate tendon span on actuator {actuator}')
        tangents = edges / lengths[:, None]
        jac = np.empty((len(sites), 3, self.model.nv))
        for j, site in enumerate(sites):
            mujoco.mj_jacSite(self.model, data, jac[j], None, int(site))
        span_jac = np.einsum('ij,ijk->ik', tangents, np.diff(jac, axis=0))
        angles = np.arctan2(
            np.linalg.norm(np.cross(tangents[:-1], tangents[1:]), axis=1),
            np.einsum('ij,ij->i', tangents[:-1], tangents[1:]),
        )
        # Inextensible tendon anchored at the tip: downstream length growth draws
        # material through a guide toward the tip (positive slip).
        span_rates = span_jac @ data.qvel
        slip = np.cumsum(span_rates[::-1])[::-1][1:]
        return span_jac, angles, slip

    def tension_ratios(self, angles: NDArray[np.float64], slip: NDArray[np.float64],
                       actuator: int) -> NDArray[np.float64]:
        """Span tension / proximal tension; the first span precedes every guide."""
        exponent = np.cumsum(self.mu[actuator] * angles * np.tanh(slip / self.velocity_eps))
        with np.errstate(over='raise', invalid='raise'):
            try:
                ratios = np.exp(np.r_[0.0, exponent])
            except FloatingPointError as exc:
                raise ValueError(f'Capstan tension overflow on actuator {actuator}') from exc
        if not np.isfinite(ratios).all():
            raise ValueError(f'Non-finite capstan tension on actuator {actuator}')
        return ratios

    def moment(self, data: mujoco.MjData, nominal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Effective actuator moment, retaining MuJoCo's negative-force pulling convention."""
        if not self.enabled:
            return nominal
        moment = nominal.copy()
        for actuator in np.flatnonzero(self.mu):
            jac, angles, slip = self.span_kinematics(data, int(actuator))
            ratios = self.tension_ratios(angles, slip, int(actuator))
            moment[actuator] = self.model.actuator_gear[actuator, 0] * (ratios @ jac)
        return moment

    def correction(self, data: mujoco.MjData, nominal: NDArray[np.float64],
                   effective: NDArray[np.float64]) -> NDArray[np.float64]:
        if not np.isfinite(data.actuator_force).all() or np.any(data.actuator_force > 0):
            raise ValueError('Capstan model requires nonpositive motor force; slack/pay-out drive is not modeled')
        return (effective - nominal).T @ data.actuator_force
