#!/bin/bash
# build mdv5 files for running in sagemaker
# usage: ./build-mdv5.sh a
#        ./build-mdv5.sh b

VERSION=$1

python lib/yolov5/export.py --weights model/md_v5${VERSION}.0.0.pt --img 640 --batch 1 --include torchscript

torch-model-archiver --model-name mdv5${VERSION} --version 1.0.0 --serialized-file model/md_v5${VERSION}.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py && mv mdv5${VERSION}.mar model/

cd model
tar czvf mdv5${VERSION}.tar.gz mdv5${VERSION}.mar
cd ../..
