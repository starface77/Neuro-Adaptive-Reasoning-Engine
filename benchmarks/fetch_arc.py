"""Fetch ARC-AGI dataset from official source.

Downloads training and evaluation challenges for ARC-AGI benchmark.
"""

import json
import urllib.request
import sys

URLS = {
    'training': 'https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/challenges.json',
    'evaluation': 'https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/challenges.json',
}

def download_arc_dataset(dataset_type='training', output_path=None):
    """Download ARC dataset."""
    if dataset_type not in URLS:
        print(f"Error: dataset_type must be 'training' or 'evaluation'")
        return False

    url = URLS[dataset_type]
    if output_path is None:
        output_path = f'arc_{dataset_type}.json'

    print(f"Downloading {dataset_type} dataset from {url}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()

        with open(output_path, 'wb') as f:
            f.write(data)

        # Validate JSON
        with open(output_path, 'r') as f:
            tasks = json.load(f)

        print(f"[OK] Downloaded {len(tasks)} tasks to {output_path}")
        return True

    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        print("\nAlternative: Clone the repository manually:")
        print("  git clone https://github.com/fchollet/ARC-AGI.git")
        print("  python benchmarks/arc_benchmark.py --dataset ARC-AGI/data/training_challenges.json")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download ARC-AGI dataset')
    parser.add_argument(
        '--type',
        choices=['training', 'evaluation', 'both'],
        default='training',
        help='Dataset type to download'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory'
    )

    args = parser.parse_args()

    success = True

    if args.type in ('training', 'both'):
        output = f"{args.output_dir}/arc_training.json"
        success = download_arc_dataset('training', output) and success

    if args.type in ('evaluation', 'both'):
        output = f"{args.output_dir}/arc_evaluation.json"
        success = download_arc_dataset('evaluation', output) and success

    sys.exit(0 if success else 1)
