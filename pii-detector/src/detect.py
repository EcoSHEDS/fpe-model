import argparse
import PIL
from detection.run_detector import load_detector
import visualization.visualization_utils as viz_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module to run a TF/PT animal detection model on lots of images')
    parser.add_argument(
        'detector_file',
        help='Path to detector model file (.pb or .pt)')
    parser.add_argument(
        'image_file',
        help='Path to a single image file, a JSON file containing a list of paths to images, or a directory')
    parser.add_argument(
        'output_file',
        help='Path to output JSON results file, should end with a .json extension')
    args = parser.parse_args()

    detector = load_detector(args.detector_file)
    image = viz_utils.load_image(args.image_file)

    # run detections on a test image to load the model
    # print('Running initial detection to load model...')
    # test_image = PIL.Image.new(mode="RGB", size=(200, 200))
    # result = detector.generate_detections_one_image(test_image, "test_image", detection_threshold=0.005)

    result = detector.generate_detections_one_image(image, "image", detection_threshold=0.005)
    print(result)
    print('\n')
