#!/bin/bash
#sudo /opt/deeplearning/install-driver.sh
conda create -n deformetrica python=3.8 numpy && source activate deformetrica
pip install deformetrica
conda activate deformetrica
