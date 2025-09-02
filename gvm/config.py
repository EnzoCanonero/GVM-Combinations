import os
import warnings
from dataclasses import dataclass
import numpy as np
import yaml

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


def build_input_data(path: str) -> input_data:
    """Parse YAML at ``path`` and return populated input_data.

    No intermediate cfg dict is exposed; everything is built directly.
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    try:
        glob = data['global']
        corr_dir = glob.get('corr_dir', '')
        name = glob['name']
        n_meas = int(glob['n_meas'])
        n_syst = int(glob['n_syst'])
    except KeyError as exc:
        raise KeyError('Global configuration must define "name", "n_meas" and "n_syst"') from exc

    try:
        combo = data['data']
        meas_entries = combo['measurements']
    except KeyError as exc:
        raise KeyError('Data configuration must define "measurements"') from exc

    labels, measurements, stat_err = [], {}, []
    for m in meas_entries:
        try:
            label = m['label']
        except KeyError as exc:
            raise KeyError('Each measurement requires a "label"') from exc
        try:
            cent = float(m['central'])
        except KeyError as exc:
            raise KeyError(f'Measurement "{label}" must define "central"') from exc
        labels.append(label)
        measurements[label] = cent
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

    try:
        syst_entries = data['syst']
    except KeyError as exc:
        raise KeyError('Configuration must define "syst" section') from exc

    meas_map = {m: i for i, m in enumerate(labels)}
    syst = {}
    corr = {}
    eoe_type = {}
    uncertain_systematics = {}

    for item in syst_entries:
        sname = item['name']
        try:
            shift_vals = item['shift']['value']
        except KeyError as exc:
            raise KeyError(f'Systematic "{sname}" must define "shift.value"') from exc
        shifts = [float(x) for x in shift_vals]

        try:
            corr_spec = item['shift']['correlation']
        except KeyError as exc:
            raise KeyError(f'Systematic "{sname}" must define "shift.correlation"') from exc
        if corr_spec == 'diagonal':
            corr_mat = np.eye(n_meas)
        elif corr_spec == 'ones':
            corr_mat = np.ones((n_meas, n_meas))
        else:
            path_corr = corr_spec.replace('${global.corr_dir}', corr_dir)
            if not os.path.isabs(path_corr):
                path_corr = os.path.join(corr_dir, path_corr)
            corr_mat = np.loadtxt(path_corr, dtype=float)

        eoe = item.get('error-on-error', {})
        eps_val = eoe.get('value', 0.0)
        eps_typ = eoe.get('type', 'dependent')
        if eps_typ not in ('dependent', 'independent'):
            raise ValueError(f'Systematic "{sname}" has unrecognised error-on-error type "{eps_typ}"')

        if eps_typ == 'independent':
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
        syst[sname] = np.array([val_map[m] for m in labels], dtype=float)
        corr[sname] = np.asarray(corr_mat, dtype=float)
        eoe_type[sname] = eps_typ

        if eps_typ == 'independent':
            eps = np.asarray(eps_val, dtype=float)
            sigma = syst[sname]
            mask = sigma != 0.0
            eps = eps[mask]
            if eps.size > 0 and np.any(eps != 0.0):
                uncertain_systematics[sname] = eps
        else:
            epsf = float(eps_val)
            if epsf != 0.0:
                uncertain_systematics[sname] = epsf

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


def validate_input_data(input_data: input_data) -> None:
    meas_names = list(input_data.measurements)
    if len(meas_names) != input_data.n_meas:
        raise ValueError(f'Expected {input_data.n_meas} measurements, got {len(meas_names)}')

    if len(input_data.syst) != input_data.n_syst:
        raise ValueError(f'Expected {input_data.n_syst} systematics, got {len(input_data.syst)}')

    if input_data.V_stat.shape != (input_data.n_meas, input_data.n_meas):
        raise ValueError(f'Stat covariance must be {input_data.n_meas}x{input_data.n_meas}')
    diff = np.argwhere(~np.isclose(input_data.V_stat, input_data.V_stat.T, rtol=1e-7, atol=1e-8))
    for i, j in diff:
        if i < j:
            warnings.warn(
                f'Stat covariance asymmetric for measurements '
                f'{meas_names[i]} and {meas_names[j]}: '
                f'{input_data.V_stat[i, j]} vs {input_data.V_stat[j, i]}')

    for name, arr in input_data.syst.items():
        if arr.shape[0] != input_data.n_meas:
            raise ValueError(f'Systematic {name} must have {input_data.n_meas} values')

    if len(input_data.corr) != input_data.n_syst:
        raise ValueError(f'Expected {input_data.n_syst} correlation matrices, got {len(input_data.corr)}')

    for name, mat in input_data.corr.items():
        if mat.shape != (input_data.n_meas, input_data.n_meas):
            raise ValueError(f'Correlation matrix {name} must be {input_data.n_meas}x{input_data.n_meas}')
        diff = np.argwhere(~np.isclose(mat, mat.T, rtol=1e-7, atol=1e-8))
        for i, j in diff:
            if i < j:
                warnings.warn(
                    f'Correlation matrix "{name}" asymmetric for measurements '
                    f'{meas_names[i]} and {meas_names[j]}: '
                    f'{mat[i, j]} vs {mat[j, i]}')
        if input_data.eoe_type.get(name, 'dependent') == 'independent':
            if not np.allclose(mat, np.eye(input_data.n_meas)):
                raise ValueError(
                    f'Systematic {name} has independent error-on-error but correlation is not diagonal')

    for name, typ in input_data.eoe_type.items():
        if typ == 'independent':
            expected = np.count_nonzero(input_data.syst[name])
            eps = np.asarray(input_data.uncertain_systematics.get(name, np.zeros(expected)))
            if eps.shape[0] != expected:
                raise ValueError(
                    f'Systematic {name} has independent error-on-error but epsilon has {eps.shape[0]} values')
        elif typ == 'dependent':
            # If provided, epsilon must be a single scalar value (not a list/vector)
            if name in input_data.uncertain_systematics:
                eps_val = input_data.uncertain_systematics[name]
                # Accept numpy scalars and Python numbers; reject lists/arrays/tuples
                if isinstance(eps_val, (list, tuple, np.ndarray)):
                    raise ValueError(
                        f'Systematic {name} has dependent error-on-error but epsilon is not a single number')
