#!/bin/bash
# fetch megadetector v5 checkpoint files

mkdir -p models/mdv5/

curl -LJO https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt
mv md_v5a.0.0.pt models/mdv5/

curl -LJO https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt
mv md_v5b.0.0.pt models/mdv5/
