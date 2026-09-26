"""Verify completed fixed-mask controls and frozen-pilot R16 SNR outputs."""
import argparse
import json
from pathlib import Path

RULE = 'snr_zero_mean_gaussian_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--repeatability', type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads((args.controls / 'snr_completion.json').read_text())
    if completion != {**completion, 'rule': RULE} or completion.get('status') != 'produced':
        raise ValueError('fixed-mask controls are incomplete or use another rule')
    checked = []
    for method in ('fmas_raw', 'dtrak_T100'):
        for track in ('gen', 'val'):
            path = args.repeatability / method / track / 'snr_statistics.json'
            payload = json.loads(path.read_text())
            if payload.get('rule') != RULE or payload.get('zeta') != 3.0:
                raise ValueError(f'R16 SNR identity differs: {method}/{track}')
            if len(payload.get('cross_reference', [])) != 16 * 16:
                raise ValueError(f'R16 8/8 per-repeat coverage differs: {method}/{track}')
            checked.append(f'{method}/{track}')
    print(json.dumps({'verdict': 'pass', 'rule': RULE, 'checked': checked}, indent=2))


if __name__ == '__main__':
    main()
