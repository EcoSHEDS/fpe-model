# Detecting Personally-Identifiable Information in Photos with MegaDetector
Photos taken at streamflow monitoring sites may contain people, vehicles, or other objects that should be screened before the photos are uploaded to Flow Photo Explorer. The MegaDetector model is an object detector trained to detect objects of 3 classes (person, vehicle, and animal) in camera trap imagery. We can use the latest versions of MegaDetector (v5) to flag images that potentially should not be uploaded.

## Setup
To detect PII in photos using MegaDetector, follow the steps outlined in the [MegaDetector repo](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).

**1. Downloading the model:**

Model checkpoints for MegaDetector v5a and MegaDetector v5b have been downloaded to the `fpe-model/results/checkpoints` directory. TBD: Should we use DVC to store these files somewhere that isn't this GitHub repo?

**2. Clone relevant repos and add them to your path within the required Python environment:**

The `ai4eutils`, `cameratraps`, and `yolov5` repos have been added as submodules in `fpe-model` and can be found under `fpe-model/third_party`.

To set up the Python environment needed for MegaDetector, use the `fpe-model/third_party/cameratraps/environment-detector.yml` file with `conda`:

```
conda env create --file environment-detector.yml
conda activate cameratraps-detector
```

Finally, add the submodule repositories to your Python path (whenever you start a new shell).
```
export PYTHONPATH="$PYTHONPATH:{FULL_PATH_TO_FPE-MODEL_REPO}/third_party/cameratraps:{FULL_PATH_TO_FPE-MODEL_REPO}/third_party/ai4eutils:{FULL_PATH_TO_FPE-MODEL_REPO}/third_party/yolov5"
```

**3. Run MegaDetector on a batch of images in a folder.**
```
python detection/run_detector_batch.py "{FULL_PATH_TO_FPE-MODEL_REPO}/results/checkpoints/md_v5a.0.0.pt" "/some/image/folder" "{FULL_PATH_TO_FPE-MODEL_REPO}/results/test_output.json" --output_relative_filenames --recursive --checkpoint_frequency 10000
```

The detection results are in the JSON file specified in the third argument to the script.