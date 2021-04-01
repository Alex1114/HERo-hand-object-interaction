#! /bin/bash

cwd=$PWD

# HERo
# https://drive.google.com/drive/u/1/folders/1tk-ATOL5s2OFST7MsohJFml2M64g4KA8
echo -e "\e[93m Download HERo model \e[0m"
cd $cwd/catkin_ws/src/hero_prediction/weight/
echo "111111" | sudo -S gdown --id 1MFD_XoiGkRl8GfpQr4JYLmOi4TssaQcr
echo "111111" | sudo -S gdown --id 1yt3Ema7eRQSWKrcB9HvSx3YeDz-uUCQT

cd $cwd/models/
echo "111111" | sudo -S gdown --id 1MFD_XoiGkRl8GfpQr4JYLmOi4TssaQcr
echo "111111" | sudo -S gdown --id 1yt3Ema7eRQSWKrcB9HvSx3YeDz-uUCQT

cd $cwd
