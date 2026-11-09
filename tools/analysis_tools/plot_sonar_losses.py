"""Plot the two sonar contrastive losses from an MMEngine text log.

Example:
    python tools/analysis_tools/plot_sonar_losses.py path/to/train.log

Only training metric lines are parsed, so configuration entries are ignored.
The plot uses the logged values without additional smoothing or rescaling.
"""

import argparse
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LOSS_NAMES = ('loss_sonar_image', 'loss_sonar_text')
TRAIN_PATTERN = re.compile(
    r'Epoch\(train\)\s*\[\s*(\d+)\s*\]'
    r'\[\s*(\d+)\s*/\s*(\d+)\s*\]')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
LOSS_PATTERNS = {
    name: re.compile(r'(?<![\w.])' + name + r':\s*(' + NUMBER + r')(?=\s|$)')
    for name in LOSS_NAMES
}


def read_losses(log_path):
    """Return independent (epoch progress, loss) samples for each metric.

    Progress is (epoch - 1) + iteration / iterations_per_epoch. For repeated
    progress values (e.g. a resumed run), the last logged value is retained.
    """
    samples = {name: {} for name in LOSS_NAMES}
    with Path(log_path).open(encoding='utf-8', errors='replace') as log_file:
        for line in log_file:
            train = TRAIN_PATTERN.search(line)
            if train is None:
                continue
            epoch, iteration, total = map(int, train.groups())
            if epoch < 1 or total < 1 or not 0 <= iteration <= total:
                continue
            progress = epoch - 1 + iteration / total
            for name, pattern in LOSS_PATTERNS.items():
                match = pattern.search(line)
                if match is not None:
                    value = float(match.group(1))
                    if math.isfinite(value):
                        samples[name][progress] = value

    missing = [name for name, values in samples.items() if not values]
    if missing:
        raise ValueError('No finite training samples found for: ' +
                         ', '.join(missing))
    return {name: sorted(values.items()) for name, values in samples.items()}


def plot_losses(samples, output, figsize=(12, 9.6), dpi=300):
    """Save both losses on a shared linear scale with automatic y limits."""
    # Match confusion_matrix.py's width:height ratio (1:0.8) and DPI.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for name, color in zip(LOSS_NAMES, ('tab:blue', 'tab:orange')):
        x, y = zip(*samples[name])
        ax.plot(x, y, label=name, color=color, linewidth=1.3)

    ax.set_title('Sonar Contrastive Losses', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training progress (epoch)', fontsize=10)
    ax.set_ylabel('Loss (logged value)', fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    ax.margins(x=0.01, y=0.08)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='MMEngine training .log file')
    parser.add_argument(
        '--out', type=Path,
        help='Output image path (default: <log_stem>_sonar_losses.png '
        'beside the log). Supports PNG, PDF and SVG.')
    parser.add_argument(
        '--figsize', nargs=2, type=float, default=(12, 9.6),
        metavar=('WIDTH', 'HEIGHT'), help='Size in inches (default: 12 9.6)')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0 or any(
            not math.isfinite(size) or size <= 0 for size in args.figsize):
        parser.error('--dpi and --figsize must be positive finite values')
    return args


def main():
    args = parse_args()
    output = args.out or args.log.with_name(
        args.log.stem + '_sonar_losses.png')
    samples = read_losses(args.log)
    plot_losses(samples, output, args.figsize, args.dpi)
    for name, values in samples.items():
        losses = [value for _, value in values]
        print(f'{name}: {len(values)} points, '
              f'epoch {values[0][0]:.4f}–{values[-1][0]:.4f}, '
              f'loss {min(losses):.6g}–{max(losses):.6g}')
    print(f'Saved: {output.resolve()}')


if __name__ == '__main__':
    main()
