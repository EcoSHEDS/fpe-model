import os
import pathlib
import argparse
import json
import copy
import pandas as pd
from tqdm import tqdm

MDV5_PII_CATEGORY_NAMES = ["person", "vehicle"]


def check_is_mdv5(detector_results_file):
    detector_results = json.load(open(detector_results_file, "r"))
    if detector_results["info"]["detector_metadata"]["megadetector_version"] not in [
        "v5a.0.0",
        "v5b.0.0",
    ]:
        raise ValueError("The detector results file is not a MegaDetector v5 file.")


def filter_mdv5_detections(detector_results, confidence_threshold=0.2, categories=[]):
    """Filter detections by confidence threshold and category.

    Args:
        detection_results: A dict containing MegaDetector v5 results.
        confidence_threshold: A float representing the confidence below
          which detections should be filtered out.
        categories: A list of categories of detections to return. Detections
          in other categories will be filtered out.

    Returns:
        A dict containing only MegaDetector v5 detection results above the
        specified confidence threshold and belonging to the specified
        categories.

    Raises:

    """
    filtered_results = copy.deepcopy(detector_results)
    for image in tqdm(filtered_results["images"]):
        # keep only detections above confidence_threshold
        # and of the specified categories
        image["detections"] = [
            det
            for det in image["detections"]
            if (det["conf"] >= confidence_threshold) and (det["category"] in categories)
        ]
        image["max_detection_conf"] = (
            max([det["conf"] for det in image["detections"]])
            if len(image["detections"]) > 0
            else 0.0
        )

    # keep only images that have at least 1 detection after filtering
    filtered_results["images"] = [
        image for image in filtered_results["images"] if len(image["detections"]) > 0
    ]
    return filtered_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-file", required=True, help="")
    parser.add_argument("--col-filename", default="filename", help="")
    parser.add_argument("--detector-results-file", required=True, help="")
    parser.add_argument("--confidence-threshold", type=float, required=True, help="")
    parser.add_argument("--output-data-file", required=True, help="")
    args = parser.parse_args()

    check_is_mdv5(
        args.detector_results_file
    )  # only MegaDetector v5 results currently supported
    detector_results = json.load(open(args.detector_results_file, "r"))
    pii_categories = [
        k
        for k, v in detector_results["detection_categories"].items()
        if v in MDV5_PII_CATEGORY_NAMES
    ]
    detector_results = filter_mdv5_detections(
        detector_results,
        confidence_threshold=args.confidence_threshold,
        categories=pii_categories,
    )
    images_with_detections = pd.DataFrame(detector_results["images"])["file"].tolist()

    df = pd.read_csv(args.data_file)
    df = df[~df[args.col_filename].isin(images_with_detections)]
    pathlib.Path(os.path.dirname(args.output_data_file)).mkdir(
        parents=True, exist_ok=True
    )
    df.to_csv(args.output_data_file)
