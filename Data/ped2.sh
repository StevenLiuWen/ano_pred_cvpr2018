#!/bin/bash

echo "Downloading UCSD-Ped2 dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mnifbgr-4ZbLb7e0dvvIoiaKFz5BdUTKRegB_vYHMYDO-BIDM0PYjIQupSNbSLFVjaZGfY9VPKS2ID5BooAqnlM5W4cmnrzsnflicxYq1H5Ne__ko4dNvrvijr4dXwJNzA0wBRN9evE0bUkct-u5VfY6pvcWtPNIPUm2NgeXpC9XFmWKG7oXL7b1-H11-C1hyho2BmWXpMqPDwo6cFqtZKA/ped2.tar.gz"
tar -xvf ped2.tar.gz
rm ped2.tar.gz

echo "Download UCSD-Ped2 successfully..."
