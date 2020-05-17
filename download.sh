#!/bin/bash
# Download dataset and model parameters
set -e

echo "Download DuReader-robust dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-data.tar.gz 
tar -zxvf dureader_robust-data.tar.gz
mv dureader_robust-data data
rm dureader_robust-data.tar.gz

wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-test1.tar.gz
tar -zxvf dureader_robust-test1.tar.gz 
mv dureader_robust-test1/test1.json data/
rm -r dureader_robust-test1/
rm dureader_robust-test1.tar.gz

# for my docker only
export LD_LIBRAYR_PATH=/opt/tiger/cuda/lib64
ln -s /opt/tiger/cuda/lib64/libcudnn.so.7 /opt/tiger/cuda/lib64/libcudnn.so