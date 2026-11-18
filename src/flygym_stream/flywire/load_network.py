"""CSV connectome loader and preflight audit for the existing hybrid runner.

The configured fast-transmitter approximation is explicitly incomplete.
Run this file directly for an audit without starting FlyGym or simulation.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order

POPULATIONS = ('input_left', 'input_right', 'output_left', 'output_right')


def normalized(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def column(frame, requested, aliases, required=True):
    if requested:
        if requested not in frame.columns:
            raise ValueError(f'Column {requested!r} missing. Available: {list(frame.columns)}')
        return requested
    keys = {normalized(a) for a in aliases}
    matches = [c for c in frame.columns if normalized(c) in keys]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise ValueError(f'Set column explicitly. Candidates: {matches}; headers: {list(frame.columns)}')


def checked_ids(series):
    result = series.astype(str).str.strip()
    if not result.str.fullmatch(r'\d+').all():
        raise ValueError('Neuron IDs must be digit strings, never rounded floats/scientific notation.')
    return result


def classify(label, cfg):
    """Resolve explicit rules first, then an opt-in, auditable sign approximation."""
    rules = {normalized(k): v for k, v in cfg['nt_class_map'].items()}
    key = normalized(label)
    if key in rules:
        return rules[key]
    if cfg['data'].get('unmapped_transmitters') != 'guess':
        return 'disabled'
    assumptions = cfg.get('transmitter_guesses', {})
    aliases = {'ach': 'exc', 'acetylcholine': 'exc', 'cholinergic': 'exc',
               'gaba': 'inh', 'gabaergic': 'inh', 'glut': 'inh', 'glutamate': 'inh',
               'hist': 'inh', 'histamine': 'inh', 'da': 'mod', 'dopamine': 'mod',
               'ser': 'mod', 'serotonin': 'mod', '5ht': 'mod', 'oct': 'mod',
               'octopamine': 'mod', 'tyr': 'mod', 'tyramine': 'mod',
               'nitricoxide': 'mod', 'no': 'mod'}
    parts = [aliases.get(normalized(x), 'unknown') for x in re.split(r'[,;+|/]', str(label))]
    fast = set(parts) & {'exc', 'inh'}
    if len(fast) > 1:
        return assumptions.get('mixed_fast_class', 'inh')
    if fast:
        return next(iter(fast))
    if 'mod' in parts:
        return assumptions.get('pure_modulator_class', 'exc')
    return assumptions.get('unknown_class', 'exc')


def load_network(cfg, folder, demo=False, audit_only=False):
    dynamics = cfg['neural'].get('dynamics', 'conductance')
    if dynamics not in ('conductance', 'shiu'):
        raise ValueError('neural.dynamics must be conductance or shiu')
    if dynamics == 'shiu':
        from .shiu_model import ShiuModel as Model
    else:
        from .neural_model import NeuralModel as Model
    if demo:
        from .neural_model import NeuralModel
        ids = ['demo_sensory_L', 'demo_sensory_R', 'demo_output_L', 'demo_output_R']
        populations = dict(input_left=[ids[0]], input_right=[ids[1]],
                           output_left=[ids[2]], output_right=[ids[3]])
        model = Model(ids, [0, 1], [2, 3], [40, 40], ['exc'] * 4, cfg['neural'])
        return model, populations, {'mode': 'SYNTHETIC DEMO', 'neurons': 4, 'edges': 2}
    folder = Path(folder)
    settings, columns = cfg['data'], cfg['columns']
    policy = settings.get('unmapped_transmitters', 'error')
    if policy not in ('error', 'disable', 'guess'):
        raise ValueError('unmapped_transmitters must be error, disable, or guess.')
    for key in ('mixed_fast_class', 'pure_modulator_class', 'unknown_class'):
        if cfg.get('transmitter_guesses', {}).get(key, 'exc') not in ('exc', 'inh'):
            raise ValueError(f'transmitter_guesses.{key} must be exc or inh')
    populations = cfg['populations']
    for name in POPULATIONS:
        values = populations[name]
        if not values or any(not isinstance(v, str) or not v.isdigit() for v in values):
            raise ValueError(f'{name} requires nonempty quoted neuron ID strings.')
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate IDs in {name}.')
    if set(populations['output_left']) & set(populations['output_right']):
        raise ValueError('Left/right output groups must be distinct for this runner.')
    attrs_path = (folder / settings['neurons_csv']).resolve()
    conn_path = (folder / settings['connections_csv']).resolve()
    print(f'Reading neuron metadata: {attrs_path}', flush=True)
    attrs = pd.read_csv(attrs_path, dtype=str, keep_default_na=False)
    idcol = column(attrs, columns.get('neuron_id'), ['root_id', 'Root ID', 'banc_888_id'])
    attrs[idcol] = checked_ids(attrs[idcol])
    if attrs[idcol].duplicated().any():
        raise ValueError('Duplicate IDs in neuron attributes.')
    original_rows = len(attrs)
    excluded_classes = {normalized(x) for x in settings.get('exclude_super_classes', [])}
    if excluded_classes and 'Super Class' not in attrs.columns:
        raise ValueError('Super Class is required for configured non-neuronal exclusions.')
    excluded = attrs['Super Class'].map(normalized).isin(excluded_classes) if excluded_classes else pd.Series(False, index=attrs.index)
    excluded_ids = set(attrs.loc[excluded, idcol])
    attrs = attrs.loc[~excluded].reset_index(drop=True)
    verified = column(attrs, columns.get('verified_nt'), ['Verified NT type', 'nt_type_verified', 'verified_nt'], False)
    predicted = column(attrs, columns.get('predicted_nt'), ['Predicted NT type', 'nt_type', 'predicted_nt'], False)
    labels = attrs[verified].str.strip() if verified else pd.Series('', index=attrs.index)
    if predicted:
        labels = labels.where(labels.ne(''), attrs[predicted].str.strip())
    rules = {normalized(k): v for k, v in cfg['nt_class_map'].items()}
    if not set(rules.values()).issubset({'exc', 'inh', 'disabled'}):
        raise ValueError('NT mappings must be exc, inh, or disabled.')
    ids = attrs[idcol].tolist()
    lookup = dict(zip(ids, range(len(ids))))
    label_classes = {label: classify(label, cfg) for label in labels.unique()}
    classes = labels.map(label_classes).to_numpy(dtype='<U8')
    inferred = ~labels.map(normalized).isin(rules) if policy == 'guess' else pd.Series(False, index=labels.index)
    for value, kind in cfg.get('neuron_class_overrides', {}).items():
        if value not in lookup or kind not in ('exc', 'inh', 'disabled'):
            raise ValueError(f'Invalid neuron override: {value} -> {kind}')
        classes[lookup[value]] = kind
        inferred.iloc[lookup[value]] = False
    for name in POPULATIONS:
        missing = set(populations[name]) - lookup.keys()
        if missing:
            raise ValueError(f'{name}: IDs absent or excluded: {sorted(missing)[:10]}')
    for name in ('input_left', 'input_right'):
        if any(classes[lookup[x]] == 'disabled' for x in populations[name]):
            raise ValueError(f'{name} contains neurons with disabled outgoing effects.')
    warnings = [
        'Point-neuron parameters and count-to-conductance scale are uncalibrated.',
        'Hybrid gait generation bypasses biological VNC-to-muscle control.',
        'Structural reachability is not evidence of neural activation or behavior.',
    ]
    if policy == 'disable':
        warnings.append('Unsupported transmitter outputs are OMITTED, not simulated as neutral physiology.')
    if policy == 'guess':
        warnings.append('Expanded sign approximation: modulators act as fast conductances; mixed labels collapse to one sign. Receptor dynamics and cotransmission are not modeled. Unknown signs are guesses.')
    print(f'{len(ids):,} retained metadata rows; {len(excluded_ids):,} non-neuronal rows excluded.', flush=True)
    print('Reading edges. Large CSVs can take several minutes.', flush=True)
    headers = pd.read_csv(conn_path, nrows=0)
    precol = column(headers, columns.get('pre'), ['pre_root_id', 'pre_pt_root_id', 'source', 'pre_id'])
    postcol = column(headers, columns.get('post'), ['post_root_id', 'post_pt_root_id', 'target', 'post_id'])
    countcol = column(headers, columns.get('count'), ['syn_count', 'synapse_count', 'weight', 'synapses'])
    pp, qq, ww = [], [], []
    read_rows = excluded_edge_rows = zero_rows = missing_edge_rows = 0
    missing_ids = set()
    for chunk in pd.read_csv(conn_path, dtype=str, keep_default_na=False,
                             usecols=[precol, postcol, countcol], chunksize=250000):
        pre, post = checked_ids(chunk[precol]), checked_ids(chunk[postcol])
        counts = pd.to_numeric(chunk[countcol], errors='coerce').to_numpy(float)
        if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError('Connection weights must be finite nonnegative integer synapse counts.')
        read_rows += len(chunk)
        zero_rows += int((counts == 0).sum())
        nonneural = pre.isin(excluded_ids) | post.isin(excluded_ids)
        excluded_edge_rows += int((nonneural & (counts > 0)).sum())
        candidate = ~nonneural & (counts > 0)
        a, b = pre.map(lookup), post.map(lookup)
        missing = candidate & (a.isna() | b.isna())
        missing_edge_rows += int(missing.sum())
        if missing.any():
            missing_ids.update(pre[missing & a.isna()])
            missing_ids.update(post[missing & b.isna()])
        valid = candidate & ~missing
        pp.append(a[valid].to_numpy(np.int64))
        qq.append(b[valid].to_numpy(np.int64))
        ww.append(counts[valid])
        if read_rows % 1000000 == 0:
            print(f'  {read_rows:,} edge rows inspected...', flush=True)
    if missing_ids:
        raise ValueError(f'{len(missing_ids):,} edge endpoint IDs lack metadata across {missing_edge_rows:,} rows. '
                         f'Examples: {sorted(missing_ids)[:8]}. Use matched exports; missing neurons are not silently dropped.')
    if not pp:
        raise ValueError('Connections CSV has no data rows.')
    pre, post, counts = np.concatenate(pp), np.concatenate(qq), np.concatenate(ww)
    del pp, qq, ww
    if not len(counts):
        raise ValueError('No positive neuronal edges remain.')
    n = len(ids)
    used = np.zeros(n, dtype=bool)
    used[pre] = True; used[post] = True
    active = classes[pre] != 'disabled'
    active_sources = np.unique(pre)
    disabled_sources = active_sources[classes[active_sources] == 'disabled']
    omitted_labels = dict(Counter(labels.iloc[disabled_sources].replace('', '(blank)')))
    # This directed graph includes inhibitory edges for structural auditing only.
    # A synthetic root connects to both sensory populations for one multi-source BFS.
    roots = np.array([lookup[x] for k in ('input_left', 'input_right') for x in populations[k]], dtype=np.int64)
    src = np.concatenate((pre[active], np.full(len(roots), n, dtype=np.int64)))
    dst = np.concatenate((post[active], roots))
    graph = coo_matrix((np.ones(len(src), dtype=bool), (src, dst)), shape=(n+1, n+1)).tocsr()
    reached = np.zeros(n+1, dtype=bool)
    reached[breadth_first_order(graph, n, directed=True, return_predecessors=False)] = True
    positive = np.bincount(post[active], minlength=n)
    outgoing = np.bincount(pre[active], minlength=n)
    pop_report = {}
    errors = []
    for name in POPULATIONS:
        idx = np.array([lookup[x] for x in populations[name]])
        entry = dict(neurons=len(idx), present_in_edges=int(used[idx].sum()),
                     with_active_incoming=int((positive[idx] > 0).sum()),
                     with_active_outgoing=int((outgoing[idx] > 0).sum()),
                     reachable_from_inputs=int(reached[idx].sum()))
        pop_report[name] = entry
        if not used[idx].all():
            errors.append(f'{name}: some selected neurons have no edges in this export.')
        if name.startswith('input') and not np.all(outgoing[idx] > 0):
            errors.append(f'{name}: some selected inputs have no active outgoing edges.')
        if name.startswith('output') and not reached[idx].all():
            errors.append(f'{name}: some outputs cannot be reached from the selected inputs under this transmitter policy.')
    if policy == 'error' and len(disabled_sources):
        errors.append(f'{len(disabled_sources):,} presynaptic neurons have disabled/unsupported transmitter effects.')
    super_counts = attrs.loc[used, 'Super Class'].value_counts().to_dict() if 'Super Class' in attrs else {}
    report = dict(mode='REAL CSV NETWORK / INCOMPLETE FAST-TRANSMITTER APPROXIMATION',
                  metadata_rows=original_rows, neurons=n, explicitly_excluded_non_neuronal_rows=len(excluded_ids),
                  neurons_with_edges=int(used.sum()), isolated_metadata_rows=int((~used).sum()),
                  raw_csv_rows=read_rows, raw_positive_rows=len(counts),
                  zero_count_rows=zero_rows, excluded_non_neuronal_edge_rows=excluded_edge_rows,
                  missing_attribute_neurons=0, active_unique_edges=int(graph.nnz-len(np.unique(roots))),
                  disabled_sources=len(disabled_sources), disabled_edge_rows=int((~active).sum()),
                  omitted_synapse_count=float(counts[~active].sum()),
                  total_neuronal_synapse_count=float(counts.sum()),
                  transmitter_class_counts=dict(Counter(classes.tolist())),
                  disabled_presynaptic_labels=omitted_labels,
                  transmitter_policy=policy,
                  inferred_synapse_count=float(counts[inferred.to_numpy()[pre]].sum()),
                  transmitter_assignments=[dict(label=str(label) or '(blank)',
                      assigned_class=label_classes[label],
                      inferred=bool(policy == 'guess' and normalized(label) not in rules),
                      metadata_neurons=int((labels == label).sum())) for label in sorted(label_classes)],
                  neuron_class_overrides=cfg.get('neuron_class_overrides', {}),
                  graph_super_class_counts=super_counts, populations=pop_report,
                  warnings=warnings, preflight_errors=errors,
                  connectivity_file=dict(path=str(conn_path), bytes=conn_path.stat().st_size),
                  metadata_file=dict(path=str(attrs_path), bytes=attrs_path.stat().st_size))
    del graph, src, dst
    if audit_only:
        return None, populations, report
    if errors:
        raise ValueError('Preflight failed: ' + ' '.join(errors) + ' Run load_network.py to save the complete report.')
    print(f'Building sparse neural matrices for {n:,} nodes...', flush=True)
    from .neural_model import NeuralModel
    model = Model(ids, pre, post, counts, classes, cfg['neural'])
    report['dynamics'] = dynamics
    report['active_unique_edges'] = model.exc.nnz + model.inh.nnz
    return model, populations, report


def main():
    parser = argparse.ArgumentParser(description='Audit real connectivity; does not start the fly.')
    parser.add_argument('--config', default=str(Path(__file__).with_name('hybrid_config.json')))
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    _, _, report = load_network(cfg, cfg_path.parent, audit_only=True)
    destination = Path(args.report) if args.report else cfg_path.with_name('network_audit.json')
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nAudit saved: {destination.resolve()}')
    if report['preflight_errors']:
        raise SystemExit('PRECHECK FAILED. Resolve the listed errors before simulation.')
    print('PRECHECK PASSED: structural checks only; physiology and behavior remain unvalidated.')


if __name__ == '__main__':
    main()
