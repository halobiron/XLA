"""Render CUB query/context pairs from a saved run, using global image IDs."""
import argparse
from pathlib import Path
from torchvision.transforms import Resize
from tacsx.analysis import render_pairs
from tacsx.data import CUB200Images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run directory containing selected_pairs.csv")
    parser.add_argument("--data-root", default="data/CUB_200_2011")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args(argv)
    run = Path(args.run)
    images = CUB200Images(args.data_root, Resize((224, 224)))
    render_pairs(images, images, str(run / "selected_pairs.csv"),
                 str(run / "retrieval_examples"), limit=args.limit)


if __name__ == "__main__":
    main()
