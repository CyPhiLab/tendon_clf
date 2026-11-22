"""Offline prediction and bounded, output-shaft velocity feedback."""

import math
import time

import mujoco
import numpy as np

from spirob_zmq.common import step_current

DIAGNOSTICS = '/spirob/velocity_feedback'
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
BAD_STATE_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                      mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)


def control_mode(node):
    mode = node.declare_parameter('motor_control_mode', 'qp_current')
    if mode not in ('qp_current', 'qp_velocity_host'):
        raise ValueError('motor_control_mode must be qp_current or qp_velocity_host; '
                         'native MIT total-current limiting is not verified')
    return mode


def positive_parameter(node, name, default):
    value = float(node.declare_parameter(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def reference_source(node):
    source = node.declare_parameter('velocity_reference_source', 'ekf_prior')
    if source not in ('ekf_prior', 'qp_rollout'):
        raise ValueError('velocity_reference_source must be ekf_prior or qp_rollout')
    return source


def check_prediction_advance(data, start_time, duration, warning_counts):
    if (any(data.warning[w].number != count for w, count in zip(BAD_STATE_WARNINGS, warning_counts))
            or not math.isclose(data.time - start_time, duration, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError('MuJoCo prediction encountered an invalid state or automatic reset')


def state_snapshot(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return {
        'dimensions': [model.nq, model.nv, model.na, model.nu, model.nmocap,
                       model.nuserdata, model.npluginstate],
        'state': state,
    }


def restore_snapshot(model, data, snapshot):
    dimensions = [model.nq, model.nv, model.na, model.nu, model.nmocap,
                  model.nuserdata, model.npluginstate]
    state = np.asarray(snapshot['state'], dtype=float)
    if (list(snapshot['dimensions']) != dimensions
            or state.shape != (mujoco.mj_stateSize(model, STATE_SPEC),)
            or not np.isfinite(state).all()):
        raise ValueError('Prediction snapshot is invalid or incompatible with the current-source model')
    mujoco.mj_setState(model, data, state, STATE_SPEC)


class VelocityFeedback:
    """Configuration is keyed by motor ID; vectors follow the QP actuator order."""

    def __init__(self, node, model):
        if node.declare_parameter('drive_model', 'current') != 'current':
            raise ValueError('qp_velocity_host currently requires drive_model=current on all nodes; '
                             'motor-drive activation/DOF projection is not supported')
        self.motor_ids = list(node.declare_parameter('motor_ids'))
        rows = node.declare_parameter('velocity_feedback_motors')
        if (len(self.motor_ids) != model.nu or len(set(self.motor_ids)) != model.nu
                or not isinstance(rows, list)
                or len(rows) != model.nu
                or {r['motor_id'] for r in rows} != set(self.motor_ids)):
            raise ValueError('velocity_feedback_motors must map every QP motor ID exactly once')
        rows = [next(r for r in rows if r['motor_id'] == mid) for mid in self.motor_ids]
        self.tendon_names = [r['tendon_name'] for r in rows]
        self.tendon_ids = np.array([model.tendon(name).id for name in self.tendon_names])
        for i, tid in enumerate(self.tendon_ids):
            if (model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON
                    or model.actuator_trnid[i, 0] != tid):
                raise ValueError(f'Motor {self.motor_ids[i]} tendon does not match QP actuator {i}')
        self.radius = np.array([r['radius_m'] for r in rows], dtype=float)
        self.sign = np.array([r['sign'] for r in rows], dtype=float)
        self.kt = np.array([r['kt_output_nm_a'] for r in rows], dtype=float)
        self.kd = np.array([r['kd_nm_s_rad'] for r in rows], dtype=float)
        self.speed_limit = np.array([r['speed_limit_rad_s'] for r in rows], dtype=float)
        if any(value.shape != (model.nu,) for value in
               (self.radius, self.sign, self.kt, self.kd, self.speed_limit)):
            raise ValueError('Each motor feedback parameter must be a scalar')
        for name, value in (('radius', self.radius), ('Kt output', self.kt),
                            ('speed limit', self.speed_limit), ('Kd', self.kd)):
            if not np.isfinite(value).all() or np.any(value < 0 if name == 'Kd' else value <= 0):
                raise ValueError(f'{name} must be finite and {"nonnegative" if name == "Kd" else "positive"}')
        if not np.isin(self.sign, [-1, 1]).all():
            raise ValueError('Motor winding signs must be -1 or +1')
        if not np.allclose(model.actuator_gear[:, 0], self.sign * self.kt / self.radius,
                           rtol=1e-3):
            raise ValueError('Winding signs, output Kt and radii disagree with model actuator gear')

    def speed_reference(self, tendon_velocity):
        raw = self.sign * np.asarray(tendon_velocity) / self.radius
        return np.clip(raw, -self.speed_limit, self.speed_limit), raw

    def ordered(self, motor_ids, values):
        values = np.asarray(values, dtype=float)
        if (len(motor_ids) != len(self.motor_ids) or set(motor_ids) != set(self.motor_ids)
                or values.shape != self.kd.shape or not np.isfinite(values).all()):
            raise ValueError('Message must contain every configured motor exactly once with finite values')
        return values[[list(motor_ids).index(mid) for mid in self.motor_ids]]

    def validate_prediction(self, prediction, motor_ids, source, now, timeout):
        if prediction['reference_source'] != source:
            raise ValueError('Velocity reference source mismatch')
        if prediction['prediction_input_kind'] != 'qp_nominal':
            raise ValueError('Velocity reference must exclude feedback current')
        velocity = self.ordered(motor_ids, prediction['tendon_velocity_m_s'])
        current = self.ordered(motor_ids, prediction['prediction_input_current_a'])
        names = prediction['tendon_names']
        if (len(names) != len(self.motor_ids)
                or [names[list(motor_ids).index(mid)] for mid in self.motor_ids] != self.tendon_names):
            raise ValueError('Velocity reference tendon mapping mismatch')
        wall_time = float(prediction['reference_wall_time'])
        stamp = float(prediction['reference_stamp'])
        if not np.isfinite([wall_time, stamp]).all() or not 0 <= now - wall_time <= timeout:
            raise ValueError('Invalid or stale velocity reference')
        step_value = float(prediction['prediction_steps'])
        if not math.isfinite(step_value) or step_value < 1 or not step_value.is_integer():
            raise ValueError('Velocity reference requires a positive-step prediction')
        steps = int(step_value)
        timing = {key: float(prediction[key]) for key in
                  ('prediction_duration_s', 'prediction_compute_s', 'reference_extract_compute_s')}
        if (not np.isfinite(list(timing.values())).all()
                or timing['prediction_duration_s'] <= 0
                or any(value < 0 for value in timing.values())):
            raise ValueError('Invalid velocity reference timing')
        return dict(prediction, motor_ids=list(self.motor_ids),
                    tendon_velocity_m_s=velocity, prediction_input_current_a=current,
                    tendon_names=list(self.tendon_names), reference_wall_time=wall_time,
                    reference_stamp=stamp, prediction_steps=steps, **timing)

    def command(self, qp_current, speed_reference, measured_speed, lower, upper):
        qp_current, speed_reference, measured_speed, lower, upper = (
            np.asarray(v, dtype=float) for v in
            (qp_current, speed_reference, measured_speed, lower, upper))
        if any(v.shape != self.kd.shape or not np.isfinite(v).all()
               for v in (qp_current, speed_reference, measured_speed, lower, upper)):
            raise ValueError('Feedback inputs must be finite per-motor vectors')
        error = speed_reference - measured_speed
        feedback_torque = self.kd * error
        requested = qp_current + feedback_torque / self.kt
        if not np.isfinite(requested).all():
            raise ValueError('Non-finite feedback command')
        # Servo encoding truncates to mA. Keep the encoded command inside even non-grid bounds.
        lo_ma, hi_ma = np.ceil(lower * 1000), np.floor(upper * 1000)
        if np.any(lo_ma > hi_ma):
            raise ValueError('Current bounds contain no representable servo command')
        sent = np.clip(np.trunc(requested * 1000), lo_ma, hi_ma) / 1000
        wire = np.trunc(sent * 1000) / 1000
        if np.any(wire < lower) or np.any(wire > upper):
            raise ValueError('Servo quantization cannot safely represent these current bounds')
        return sent, {
            'qp_current_a': qp_current,
            'feedforward_torque_nm': self.kt * qp_current,
            'speed_reference_rad_s': speed_reference,
            'measured_speed_rad_s': measured_speed,
            'velocity_error_rad_s': error,
            'feedback_torque_nm': feedback_torque,
            'feedback_torque_kind': 'host_requested',
            'requested_current_a': requested,
            'requested_torque_nm': self.kt * requested,
            'sent_current_a': wire,
            'sent_torque_nm': self.kt * wire,
            'saturated': (requested < lower) | (requested > upper),
            'current_lower_a': lower,
            'current_upper_a': upper,
        }


class TendonPredictor:
    def __init__(self, model, config, period, vlimit, friction=None):
        self.model, self.config = model, config
        self.data = mujoco.MjData(model)
        self.vlimit, self.friction = vlimit, friction
        dt = float(model.opt.timestep)
        if not math.isfinite(period) or period <= 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Prediction period and model timestep must be finite and positive')
        self.steps = max(1, int(math.floor(period / dt + 0.5)))
        self.duration = self.steps * dt

    def predict(self, source, current):
        started = time.perf_counter()
        current = np.asarray(current, dtype=float)
        if current.shape != (self.model.nu,) or not np.isfinite(current).all():
            raise ValueError('Prediction current must be a finite vector in QP order')
        # Node callbacks are serialized; no estimator data or shared MjData is advanced here.
        mujoco.mj_copyData(self.data, self.model, source)
        warning_counts = [self.data.warning[w].number for w in BAD_STATE_WARNINGS]
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_fwdVelocity(self.model, self.data)
        step_current(self.model, self.data, current, self.steps, self.vlimit, self.friction)
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_fwdVelocity(self.model, self.data)
        check_prediction_advance(self.data, source.time, self.duration, warning_counts)
        velocity = self.data.ten_velocity[self.config.tendon_ids].copy()
        if not np.isfinite(velocity).all():
            raise ValueError('Non-finite predicted tendon velocity')
        reference, raw = self.config.speed_reference(velocity)
        return reference, {
            'tendon_velocity_m_s': velocity,
            'unclipped_speed_reference_rad_s': raw,
            'speed_saturated': raw != reference,
            'prediction_steps': self.steps,
            'prediction_duration_s': self.duration,
            'prediction_compute_s': time.perf_counter() - started,
        }
