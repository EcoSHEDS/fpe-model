# python src/detect-s3.py model/md_v5a.0.0.pt sagemaker-us-east-1-474916309046 "fpe-pii/imagesets/atherton/img/Atherton Brook__2023-02-15__13-48-48(49).JPG" "fpe-pii/imagesets/atherton/pii/Atherton Brook__2023-02-15__13-48-48(49).json"
import argparse
import boto3
import PIL
import json
from io import BytesIO, StringIO
from detection.run_detector import load_detector

s3 = boto3.client('s3')

def read_image_from_s3(bucket_name, image_key):
    # Read the image from the S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key=image_key)
    image_data = response['Body'].read()

    # Load the image using the PIL (Python Imaging Library) Image module
    image = PIL.Image.open(BytesIO(image_data))

    return image

def save_object_to_s3_json(bucket_name, object_key, data):
    # Convert the data to a JSON string
    json_data = json.dumps(data)

    # Save the JSON string to the S3 bucket
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=StringIO(json_data).getvalue())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module to run a TF/PT animal detection model on lots of images')
    parser.add_argument(
        'bucket',
        help='S3 bucket name')
    parser.add_argument(
        'input_key',
        help='S3 key to input image')
    parser.add_argument(
        'output_key',
        help='S3 key to output JSON results file, should end with a .json extension'
    )
    args = parser.parse_args()

    detector = load_detector('model/md_v5a.0.0.pt')

    image = read_image_from_s3(args.bucket, args.input_key)
    result = detector.generate_detections_one_image(image, "image", detection_threshold=0.005)
    print(result)
    print('\n')

    save_object_to_s3_json(args.bucket, args.output_key, result)
