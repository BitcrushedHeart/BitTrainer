"""Write a ConvNeXt V2 trunk converted from Apache-2.0 ConvNeXt V1 weights.

    python -m bittrainer.tools.convnext_v1_trunk --size base --out base_trunk.safetensors

Downloads the V1 weights, converts them (see ``bittrainer.convnext_v1_init``),
checks that the converted trunk gives the V1 model's features, and saves it.
Train from it with ``backbone_pack_train --init-checkpoint`` (or let that tool
do all of this itself with ``--init apache``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bittrainer.convnext_v1_init import V1_WEIGHTS, write_v1_trunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.convnext_v1_trunk",
        description="Convert Apache-2.0 ConvNeXt V1 weights into a ConvNeXt V2 trunk.",
    )
    parser.add_argument("--size", required=True, choices=sorted(V1_WEIGHTS))
    parser.add_argument("--out", required=True, help="the safetensors file to write")
    parser.add_argument(
        "--weights", default=None, help="another timm ConvNeXt V1 weight name"
    )
    args = parser.parse_args(argv)

    try:
        info = write_v1_trunk(args.size, Path(args.out), args.weights)
    except (ValueError, RuntimeError) as refusal:
        print(f"[trunk] REFUSED: {refusal}", flush=True)
        return 2
    print(
        f"[trunk] {info['model_id']} ({info['license']}) converted; features match "
        f"to {info['max_abs_difference']:.2g}. Written to {info['path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
