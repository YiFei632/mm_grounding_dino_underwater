"""Export loss_sonar_class training samples from an MMEngine log to CSV.

The ``epoch`` column is continuous training progress, computed from the
logged epoch, iteration, and iterations per epoch. This distributes samples
through the 12 epochs (or however many epochs are present in the log), with
fractional epoch values between epoch boundaries.

Example:
    python tools/analysis_tools/export_sonar_class_loss.py path/to/train.log
"""

import argparse
import csv
import math
from pathlib import Path
import re


TRAIN_PATTERN = re.compile(
    r'Epoch\(train\)\s*\[\s*(\d+)\s*\]'
    r'\[\s*(\d+)\s*/\s*(\d+)\s*\]')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
LOSS_PATTERN = re.compile(
    r'(?<![\w.])loss_sonar_class:\s*(' + NUMBER + r')(?=\s|$)')


def read_samples(log_path):
    """Parse (epoch progress, loss_sonar_class) from training records."""
    samples = []
    with Path(log_path).open(encoding='utf-8', errors='replace') as log_file:
        for line in log_file:
            train = TRAIN_PATTERN.search(line)
            if train is None:
                continue
            loss = LOSS_PATTERN.search(line)
            if loss is None:
                continue

            epoch, iteration, total = map(int, train.groups())
            if epoch < 1 or total < 1 or not 0 <= iteration <= total:
                continue
            value = float(loss.group(1))
            if not math.isfinite(value):
                continue

            # Epoch 1 occupies [0, 1], epoch 2 [1, 2], etc.
            epoch_progress = (epoch - 1) + iteration / total
            samples.append((epoch_progress, value))

    if not samples:
        raise ValueError('No loss_sonar_class training samples found in ' +
                         str(log_path))
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='MMEngine training .log file')
    parser.add_argument(
        '--out', type=Path,
        help='Output CSV path (default: <log_stem>_loss_sonar_class.csv '
        'beside the log).')
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.out or args.log.with_name(
        args.log.stem + '_loss_sonar_class.csv')
    samples = read_samples(args.log)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('epoch', 'loss_sonar_class'))
        writer.writerows((f'{epoch:.8f}'.rstrip('0').rstrip('.'),
                          f'{loss:.10g}') for epoch, loss in samples)

    print(f'Exported {len(samples)} samples to {output.resolve()}')
    print(f'Epoch range: {samples[0][0]:.4f}–{samples[-1][0]:.4f}')


if __name__ == '__main__':
    main()
