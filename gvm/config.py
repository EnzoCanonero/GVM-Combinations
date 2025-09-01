import os
import warnings
from dataclasses import dataclass
import numpy as np
import yaml


#####
# Legacy loader removed. Use load_input_data(path) to construct input data
# directly from YAML path.
#####


@dataclass
class input_data:
    name: str
    n_meas: int
    n_syst: int
    labels: list
    measurements: dict
    V_stat: np.ndarray
    syst: dict
    corr: dict
    eoe_type: dict
    uncertain_systematics: dict


def build_input_data(cfg: dict) -> input_data:
    glob = cfg['global']
    name = glob['name']
    n_meas = glob['n_meas']
    n_syst = glob['n_syst']

    measurements = {m: float(v) for m, v in cfg['data']['measurements'].items()}
    labels = list(measurements.keys())
    V_stat = np.asarray(cfg['data']['V_stat'], dtype=float)

    syst = {
        sname: np.array([cfg['syst'][sname]['values'][m] for m in measurements], dtype=float)
        for sname in cfg['syst']
    }
    corr = {s: np.asarray(cfg['syst'][s]['corr'], dtype=float) for s in cfg['syst']}
    eoe_type = {s: cfg['syst'][s]['error-on-error']['type'] for s in cfg['syst']}

    uncertain_systematics = {}
    for s in cfg['syst']:
        val = cfg['syst'][s]['error-on-error']['value']
        if eoe_type.get(s, 'dependent') == 'independent':
            eps = np.asarray(val, dtype=float)
            sigma = syst[s]
            mask = sigma != 0.0
            eps = eps[mask]
            if eps.size > 0 and np.any(eps != 0.0):
                uncertain_systematics[s] = eps
        else:
            eps = float(val)
            if eps != 0.0:
                uncertain_systematics[s] = eps

    return input_data(
        name=name,
        n_meas=n_meas,
        n_syst=n_syst,
        labels=labels,
        measurements=measurements,
        V_stat=V_stat,
        syst=syst,
        corr=corr,
        eoe_type=eoe_type,
        uncertain_systematics=uncertain_systematics,
    )


def validate_input_data(state: input_data) -> None:
    meas_names = list(state.measurements)
    if len(meas_names) != state.n_meas:
        raise ValueError(f'Expected {state.n_meas} measurements, got {len(meas_names)}')

    if len(state.syst) != state.n_syst:
        raise ValueError(f'Expected {state.n_syst} systematics, got {len(state.syst)}')

    if state.V_stat.shape != (state.n_meas, state.n_meas):
        raise ValueError(f'Stat covariance must be {state.n_meas}x{state.n_meas}')
    diff = np.argwhere(~np.isclose(state.V_stat, state.V_stat.T, rtol=1e-7, atol=1e-8))
    for i, j in diff:
        if i < j:
            warnings.warn(
                f'Stat covariance asymmetric for measurements '
                f'{meas_names[i]} and {meas_names[j]}: '
                f'{state.V_stat[i, j]} vs {state.V_stat[j, i]}')

    for name, arr in state.syst.items():
        if arr.shape[0] != state.n_meas:
            raise ValueError(f'Systematic {name} must have {state.n_meas} values')

    if len(state.corr) != state.n_syst:
        raise ValueError(f'Expected {state.n_syst} correlation matrices, got {len(state.corr)}')

    for name, mat in state.corr.items():
        if mat.shape != (state.n_meas, state.n_meas):
            raise ValueError(f'Correlation matrix {name} must be {state.n_meas}x{state.n_meas}')
        diff = np.argwhere(~np.isclose(mat, mat.T, rtol=1e-7, atol=1e-8))
        for i, j in diff:
            if i < j:
                warnings.warn(
                    f'Correlation matrix "{name}" asymmetric for measurements '
                    f'{meas_names[i]} and {meas_names[j]}: '
                    f'{mat[i, j]} vs {mat[j, i]}')
        if state.eoe_type.get(name, 'dependent') == 'independent':
            if not np.allclose(mat, np.eye(state.n_meas)):
                raise ValueError(
                    f'Systematic {name} has independent error-on-error but correlation is not diagonal')

    for name, typ in state.eoe_type.items():
        if typ == 'independent':
            expected = np.count_nonzero(state.syst[name])
            eps = np.asarray(state.uncertain_systematics.get(name, np.zeros(expected)))
            if eps.shape[0] != expected:
                raise ValueError(
                    f'Systematic {name} has independent error-on-error but epsilon has {eps.shape[0]} values')


def load_input_data(path: str) -> input_data:
    """Parse YAML at ``path`` and return populated input_data."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    cfg = {}
    try:
        glob = data['global']
        corr_dir = glob.get('corr_dir', '')
        name = glob['name']
        n_meas = int(glob['n_meas'])
        n_syst = int(glob['n_syst'])
    except KeyError as exc:
        raise KeyError('Global configuration must define "name", "n_meas" and "n_syst"') from exc

    cfg['global'] = {
        'name': name,
        'n_meas': n_meas,
        'n_syst': n_syst,
    }

    try:
        combo = data['data']
        meas_entries = combo['measurements']
    except KeyError as exc:
        raise KeyError('Data configuration must define "measurements"') from exc

    labels, central, stat_err = [], [], []
    for m in meas_entries:
        try:
            label = m['label']
        except KeyError as exc:
            raise KeyError('Each measurement requires a "label"') from exc
        try:
            cent = m['central']
        except KeyError as exc:
            raise KeyError(f'Measurement "{label}" must define "central"') from exc
        labels.append(label)
        central.append(float(cent))
        if 'stat_error' in m:
            stat_err.append(float(m['stat_error']))

    stat_cov_path = combo.get('stat_cov_path')
    if stat_cov_path:
        stat_cov_path = stat_cov_path.replace('${global.corr_dir}', corr_dir)
        if not os.path.isabs(stat_cov_path):
            stat_cov_path = os.path.join(corr_dir, stat_cov_path)
        V_stat = np.loadtxt(stat_cov_path, dtype=float)
    elif stat_err:
        V_stat = np.diag(np.array(stat_err, dtype=float) ** 2)
    else:
        raise KeyError('Measurement stat errors or covariance required')

    # Store measurements as {label: central}
    meas_data = {lab: c for lab, c in zip(labels, central)}
    cfg['data'] = {'measurements': meas_data, 'V_stat': V_stat}

    try:
        syst_entries = data['syst']
    except KeyError as exc:
        raise KeyError('Configuration must define "syst" section') from exc

    meas_map = {m: i for i, m in enumerate(labels)}
    syst_dict = {}
    for item in syst_entries:
        name = item['name']

        try:
            shift = item['shift']
            shift_vals = shift['value']
        except KeyError as exc:
            raise KeyError(f'Systematic "{name}" must define "shift.value"') from exc
        shifts = [float(x) for x in shift_vals]

        try:
            corr_spec = shift['correlation']
        except KeyError as exc:
            raise KeyError(f'Systematic "{name}" must define "shift.correlation"') from exc
        if corr_spec == 'diagonal':
            corr = np.eye(n_meas)
        elif corr_spec == 'ones':
            corr = np.ones((n_meas, n_meas))
        else:
            path_corr = corr_spec.replace('${global.corr_dir}', corr_dir)
            if not os.path.isabs(path_corr):
                path_corr = os.path.join(corr_dir, path_corr)
            corr = np.loadtxt(path_corr, dtype=float)

        eoe = item.get('error-on-error', {})
        eps_val = eoe.get('value', 0.0)
        eps_type = eoe.get('type', 'dependent')
        if eps_type not in ('dependent', 'independent'):
            raise ValueError(f'Systematic "{name}" has unrecognised error-on-error type "{eps_type}"')
        if eps_type == 'independent':
            if isinstance(eps_val, (list, tuple, np.ndarray)):
                eps_list = [float(x) for x in eps_val]
            else:
                eps_list = [float(eps_val)]
            if len(eps_list) == 1:
                eps_list *= n_meas
            eps_val = eps_list
        else:
            eps_val = float(eps_val)

        val_map = {lab: shifts[meas_map[lab]] for lab in labels}
        syst_dict[name] = {
            'values': val_map,
            'error-on-error': {'value': eps_val, 'type': eps_type},
            'corr': corr,
        }

    cfg['syst'] = syst_dict
    # Build and return input_data
    return build_input_data(cfg)


def input_data_to_cfg(state: input_data) -> dict:
    meas = list(state.labels)
    cfg = {
        'global': {
            'name': state.name,
            'n_meas': state.n_meas,
            'n_syst': state.n_syst,
        },
        'data': {
            'measurements': {
                m: {
                    'central': float(state.measurements[m]),
                    'stat': float(np.sqrt(state.V_stat[i, i]))
                }
                for i, m in enumerate(meas)
            },
            'V_stat': state.V_stat.copy(),
        },
        'syst': {
            sname: {
                'shift': {
                    'value': {
                        m: float(state.syst[sname][i])
                        for i, m in enumerate(meas)
                    },
                    'correlation': state.corr[sname].copy(),
                },
                'error-on-error': {
                    'value': (lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
                             (state.uncertain_systematics.get(sname, 0.0)),
                    'type': state.eoe_type[sname],
                },
            }
            for sname in state.syst
        },
    }
    return cfg


def apply_update(state: input_data, info: dict) -> bool:
    changed = False
    idx = {m: i for i, m in enumerate(state.measurements)}

    data = info.get('data', {})
    meas_info = data.get('measurements', {})
    for m, vals in meas_info.items():
        if 'central' in vals:
            state.measurements[m] = float(vals['central'])
            changed = True
        if 'stat' in vals:
            state.V_stat[idx[m], idx[m]] = float(vals['stat']) ** 2
            changed = True

    if 'V_stat' in data:
        V_stat = np.asarray(data['V_stat'], dtype=float)
        state.V_stat = V_stat
        changed = True

    syst_info = info.get('syst', {})
    for sname, entry in syst_info.items():
        if sname not in state.syst:
            continue

        shift = entry.get('shift')
        if shift:
            if 'value' in shift:
                for m, val in shift['value'].items():
                    state.syst[sname][idx[m]] = float(val)
                    changed = True
            if 'correlation' in shift:
                mat = np.asarray(shift['correlation'], dtype=float)
                state.corr[sname] = mat
                changed = True

        if 'error-on-error' in entry:
            eoe = entry['error-on-error']
            if 'type' in eoe:
                if eoe['type'] not in ('dependent', 'independent'):
                    raise ValueError(
                        f'Systematic "{sname}" has unrecognised error-on-error type "{eoe["type"]}"'
                    )
                state.eoe_type[sname] = eoe['type']
                changed = True
            if 'value' in eoe:
                val = eoe['value']
                if state.eoe_type.get(sname, 'dependent') == 'independent':
                    if isinstance(val, (list, tuple, np.ndarray)):
                        eps = np.asarray(val, dtype=float)
                        if eps.size == 1:
                            eps = np.repeat(eps, state.n_meas)
                        elif eps.size != state.n_meas:
                            raise ValueError(
                                f'Systematic "{sname}" epsilon has {eps.size} values, expected {state.n_meas}')
                    else:
                        eps = np.repeat(float(val), state.n_meas)
                    mask = state.syst[sname] != 0.0
                    eps = eps[mask]
                    if eps.size == 0 or np.all(eps == 0):
                        state.uncertain_systematics.pop(sname, None)
                    else:
                        state.uncertain_systematics[sname] = eps
                    changed = True
                else:
                    e = float(val)
                    if e == 0:
                        state.uncertain_systematics.pop(sname, None)
                    else:
                        state.uncertain_systematics[sname] = e
                    changed = True

    return changed
