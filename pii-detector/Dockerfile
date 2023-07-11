# FROM conda/miniconda3:latest
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install humanfriendly \
                pandas \
                matplotlib \
                seaborn==0.11.0 \
                PyYAML==6.0 \
                jsonpickle \
                opencv-python \
                boto3

RUN mkdir /app
WORKDIR /app

COPY ./lib /app/lib
COPY model/md_v5a.0.0.pt /app/model/md_v5a.0.0.pt
COPY src/*.py /app/
ENV PYTHONPATH=/app/lib/yolov5:/app/lib/cameratraps

# RUN conda env create --file /app/lib/cameratraps/environment-detector-mac.yml
# RUN conda init bash
# RUN echo "conda activate cameratraps-detector" >> ~/.bashrc

ENTRYPOINT [ "python" ]
CMD [ "detect-s3.py" ]

# docker build -t fpe-pii .
# docker run -it --rm -v data/atherton/img:/workspace/img -v data/atherton/out:/workspace/out fpe-pii:latest bash
# python lib/cameratraps/detection/run_detector_batch.py model/md_v5a.0.0.pt img/ out/out.json --output_relative_filenames --recursive