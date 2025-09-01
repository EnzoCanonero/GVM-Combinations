import os
import numpy as np
import yaml


def load_config(path: str) -> dict:
    """Parse a YAML configuration file into the internal cfg dict.

    The returned dictionary has keys: 'global', 'data', and 'syst', matching
    the structure expected by GVMCombination.__init__(cfg).
    """
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
        raise KeyError(
            'Global configuration must define "name", "n_meas" and "n_syst"'
        ) from exc

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
            raise KeyError(
                f'Systematic "{name}" must define "shift.value"'
            ) from exc
        shifts = [float(x) for x in shift_vals]

        try:
            corr_spec = shift['correlation']
        except KeyError as exc:
            raise KeyError(
                f'Systematic "{name}" must define "shift.correlation"'
            ) from exc
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
            raise ValueError(
                f'Systematic "{name}" has unrecognised error-on-error type "{eps_type}"'
            )
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
    return cfg

