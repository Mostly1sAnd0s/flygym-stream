"""Controlled neural pathway tests; no physiology is fitted or changed.

Odor is sampled at fixed antenna proxies around the origin. This isolates the
neural pathway from body feedback; use run_hybrid.py for embodied validation.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .load_network import load_network
from .neural_model import MotorDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('hybrid_config.json')))
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu',
                        help='Diagnostic backend (CPU default avoids GPU instrumentation startup); simulation runners default to GPU')
    parser.add_argument('--seconds', type=float, default=0.5)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    config = Path(args.config).resolve()
    cfg = json.loads(config.read_text(encoding='utf-8-sig'))
    cfg['neural']['backend'] = args.backend
    brain, populations, audit = load_network(cfg, config.parent)
    output = config.parent / 'Outputs' / ('diagnostic_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    xp = brain.xp
    names = ['input_left', 'input_right', 'output_left', 'output_right']
    groups = [xp.asarray(brain.indices(populations[name])) for name in names]
    decoder = MotorDecoder(brain, populations, cfg['decoder'])
    source = np.asarray(cfg['stimulus']['source_xy_mm'])
    antennae = np.array([[0., 0.25], [0., -0.25]])
    odor = np.exp(-np.sum((antennae-source)**2, axis=1)/(2*cfg['stimulus']['odor_sigma_mm']**2))
    summary = {'fixed_antenna_proxies_mm': antennae.tolist(), 'odor_levels': odor.tolist(), 'trials': {}}
    for trial in ['no_input', 'odor', 'direct_output']:
        brain.v.fill(cfg['neural']['rest_mV'])
        brain.gate.fill(0); brain.refractory_ms.fill(0); brain.rates_hz.fill(0)
        current = xp.zeros(len(brain.ids))
        if trial == 'odor':
            for group, level in zip(groups[:2], odor):
                current[group] += cfg['stimulus']['baseline_pA'] + cfg['stimulus']['amplitude_pA']*level
        if trial == 'direct_output':
            for group in groups[2:]:
                current[group] += 700.0
        counts = xp.zeros(4, dtype=np.int64)
        peak_voltage = xp.full(4, -np.inf)
        peak_ge = xp.zeros(4); peak_gi = xp.zeros(4)
        with (output / (trial + '.csv')).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'population', 'mean_v_mV', 'mean_ge_nS', 'mean_gi_nS', 'mean_rate_hz'])
            for step in range(int(np.ceil(args.seconds/0.0001))):
                spikes = brain.step(current, 0.0001)
                counts += xp.stack([spikes[g].sum() for g in groups])
                if step % 100 == 0 or step == int(np.ceil(args.seconds/0.0001))-1:
                    # Snapshot of conductances for the next step, including newly released gates.
                    ge, gi = brain.exc @ brain.gate, brain.inh @ brain.gate
                    values = xp.stack([xp.stack([brain.v[g].mean(), ge[g].mean(), gi[g].mean(), brain.rates_hz[g].mean()]) for g in groups])
                    peak_voltage = xp.maximum(peak_voltage, values[:, 0])
                    peak_ge = xp.maximum(peak_ge, values[:, 1]); peak_gi = xp.maximum(peak_gi, values[:, 2])
                    for name, row in zip(names, brain.to_cpu(values)):
                        writer.writerow([(step+1)*0.0001, name, *row])
            brain.check_health()
        drive, rates = decoder.read(brain.rates_hz)
        result = dict(spikes=dict(zip(names, brain.to_cpu(counts).tolist())),
                      sampled_peak_mean_voltage_mV=brain.to_cpu(peak_voltage).tolist(),
                      sampled_peak_mean_ge_nS=brain.to_cpu(peak_ge).tolist(),
                      sampled_peak_mean_gi_nS=brain.to_cpu(peak_gi).tolist(),
                      final_output_rates_hz=rates.tolist(), final_drive=drive.tolist())
        summary['trials'][trial] = result
        print(trial, json.dumps(result), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    main()
