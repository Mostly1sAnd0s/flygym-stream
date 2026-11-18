"""Plot recorded neural trials; this is not a body-animation replay."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / 'Outputs/eon_verification.json').read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    colors = {'sensory': '#0072B2', 'no-input': '#777777',
              'direct-output': '#D55E00', 'disconnect-output': '#009E73'}
    for condition, run in report['runs'].items():
        path = root.parent / run['path'].replace('\\', '/')
        rows = np.genfromtxt(path / 'trace.csv', delimiter=',', names=True)
        axes[0].plot(rows['time_s'] * 1000, rows['total_spikes'],
                     label=condition, color=colors[condition],
                     linestyle='--' if condition == 'disconnect-output' else '-')
        if condition == 'sensory':
            for key, label in [('steer_left_hz', 'Left steering'),
                               ('steer_right_hz', 'Right steering'),
                               ('forward_hz', 'Forward readout')]:
                axes[1].step(rows['time_s'] * 1000, rows[key], where='pre', label=label)
            for key, label in [('drive_left', 'Left leg drive'), ('drive_right', 'Right leg drive')]:
                axes[2].step(rows['time_s'] * 1000, rows[key], where='pre', label=label)
    axes[0].set_title('Recorded CPU neural trials — 157,789 neurons, seed 0\n'
                      'Prescribed input; no body motion was recorded in these trials')
    axes[0].set_ylabel('Spikes per 15 ms block')
    axes[1].set_ylabel('Sensory-trial readout (Hz)')
    axes[2].set_ylabel('Sensory-trial command')
    axes[2].set_xlabel('Simulated time (ms)')
    axes[2].set_xlim(0, 150)
    for ax in axes:
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=.2)
    destination = root / 'Outputs/eon_neural_results.png'
    fig.savefig(destination, dpi=160)
    print(destination)


if __name__ == '__main__':
    main()
