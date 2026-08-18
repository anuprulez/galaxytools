#!/usr/bin/env python3
"""Galaxy-facing command-line interface for FiBar SEM diameter prediction."""

import argparse
import csv
import random

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fibar_module.classical_segmenter import classical_segment
from fibar_module.dm_finder import dm_finder
from fibar_module.point_picker import point_picker
from fibar_module.scale_obtain import load_digit_models, scale_obtain
from fibar_module.thinner import thinner, thinner_2k_5k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--resnet-model", required=True)
    parser.add_argument("--vgg-model", required=True)
    parser.add_argument("--backend", choices=("resnet", "vgg"), default="resnet")
    parser.add_argument("--magnification", choices=("2k", "5k", "other"), default="other")
    parser.add_argument("--measurements", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--segmented-image", required=True)
    parser.add_argument("--histogram", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.measurements < 1:
        raise ValueError("--measurements must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    source = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise ValueError("Input could not be read as a TIFF image")

    load_digit_models(args.resnet_model, args.vgg_model, args.backend)
    scales = scale_obtain(args.input)
    unit = "px"
    units_per_pixel = 1.0
    if scales and len(scales) == 3 and scales[0] and scales[2]:
        if scales[1] == "um":
            units_per_pixel = float(scales[0]) * 1000.0 / float(scales[2])
            unit = "nm"
        elif scales[1] == "nm":
            units_per_pixel = float(scales[0]) / float(scales[2])
            unit = "nm"

    if args.magnification in ("2k", "5k"):
        segmented = cv2.threshold(
            source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        distance, skeleton = thinner_2k_5k(segmented)
    else:
        segmented = classical_segment(args.input)
        distance, skeleton = thinner(segmented)

    points = point_picker(segmented, args.measurements)
    height, width = segmented.shape[:2]
    diameters, exceptions, coordinates = dm_finder(
        skeleton, distance, segmented, points, height, width, units_per_pixel
    )

    segmented_encoded, segmented_buffer = cv2.imencode(".tiff", segmented)
    if not segmented_encoded:
        raise RuntimeError("Could not encode segmented TIFF output")
    with open(args.segmented_image, "wb") as handle:
        handle.write(segmented_buffer.tobytes())

    annotated = cv2.imread(args.input, cv2.IMREAD_COLOR)
    for x0, y0, xmid, ymid in coordinates:
        start = (int(y0), int(x0))
        end = (int(2 * ymid - y0), int(2 * xmid - x0))
        cv2.line(annotated, start, end, (0, 0, 0), 2)
        cv2.circle(annotated, start, 2, (255, 0, 0), -1)
        cv2.circle(annotated, end, 2, (255, 0, 0), -1)
    # Galaxy dataset paths generally have no filename extension, while imwrite
    # chooses its codec from the extension. Encode explicitly as TIFF instead.
    encoded, buffer = cv2.imencode(".tiff", annotated)
    if not encoded:
        raise RuntimeError("Could not encode annotated TIFF output")
    with open(args.image, "wb") as handle:
        handle.write(buffer.tobytes())

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(diameters, bins="auto", edgecolor="black")
    axis.set_xlabel("Diameter (%s)" % unit)
    axis.set_ylabel("Frequency")
    axis.set_title("FiBar diameter distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.histogram, format="png", dpi=150)
    plt.close(figure)

    with open(args.table, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("measurement", "diameter", "unit", "start_x", "start_y", "end_x", "end_y"))
        for index, (diameter, coord) in enumerate(zip(diameters, coordinates), 1):
            x0, y0, xmid, ymid = coord
            writer.writerow((index, diameter, unit, y0, x0, 2 * ymid - y0, 2 * xmid - x0))

    print("Scale:", scales)
    print("Successful measurements:", len(diameters))
    print("Skipped measurements:", len(exceptions))


if __name__ == "__main__":
    main()
