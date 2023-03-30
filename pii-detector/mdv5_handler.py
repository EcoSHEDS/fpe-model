"""Custom TorchServe model handler for YOLOv5 models.
"""
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import base64
import torch
import torchvision
import io
from PIL import Image

from utils.augmentations import letterbox
from utils.general import non_max_suppression, xyxy2xywh, scale_coords
import ct_utils

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    img_size = 640
    min_conf_thresh = 0.005
    stride = 64

    """Image size (px). Images will be resized to this resolution before inference.
    """

    def __init__(self):
        # call superclass initializer
        super().__init__()

    def preprocess(self, data):
        """Converts input images to float tensors.
        Args:
            data (List): Input data from the request in the form of a list of image tensors.
        Returns:
            Tensor: single Tensor of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        """
        images = []

        # load images
        # taken from https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py

        # handle if images are given in base64, etc.
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            # force convert to tensor
            # and resize to [img_size, img_size]

            target_size = self.img_size
            print(f"target_size: {target_size}")

            image = np.asarray(image)
            self.original_shape = image.shape
            print(f"original_shape: {self.original_shape}")

            img = letterbox(image, new_shape=target_size,
                                 stride=self.stride, auto=False)[0]  # JIT requires auto=False
            #print(f"img.shape: {img.shape}")
            img = img.transpose((2, 0, 1))  # HWC to CHW; PIL Image is RGB already
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            # img = img.to(self.device)
            img = img.float()
            img /= 255
            print(f"img.size(): {img.size()}")
            print(f"img.mean: {img.mean()}")

            images.append(img)

        # convert list of equal-size tensors to single stacked tensor
        # has shape BATCH_SIZE x 3 x IMG_SIZE x IMG_SIZE
        images_tensor = torch.stack(images).to(self.device)
        self.final_shape = images_tensor.numpy().shape
        print(f"final_shape: {self.final_shape}")

        return images_tensor

    def postprocess(self, inference_output):
        # perform NMS (nonmax suppression) on model outputs
        pred = non_max_suppression(inference_output[0], conf_thres=self.min_conf_thresh)

        # initialize empty list of detections for each image
        result = {}
        detections = []
        max_conf = 0.0

        CONF_DIGITS = 3
        COORD_DIGITS = 4

        gn = torch.tensor(self.original_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # This is a loop over detection batches, which will always be length 1 in our case,
        # since we're not doing batch inference.
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.final_shape[2:], det[:, :4], self.original_shape).round()

                for *xyxy, conf, cls in reversed(det):
                    # normalized center-x, center-y, width and height
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    api_box = ct_utils.convert_yolo_to_xywh(xywh)

                    conf = ct_utils.truncate_float(conf.tolist(), precision=CONF_DIGITS)

                    # MegaDetector output format's categories start at 1, but this model's start at 0
                    cls = int(cls.tolist()) + 1
                    if cls not in (1, 2, 3):
                        raise KeyError(f'{cls} is not a valid class.')

                    detections.append({
                        'category': str(cls),
                        'conf': conf,
                        'bbox': ct_utils.truncate_float_array(api_box, precision=COORD_DIGITS)
                    })
                    max_conf = max(max_conf, conf)
        result['detections'] = detections
        result['max_conf'] = max_conf
        return [result]
