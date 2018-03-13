#!/bin/bash

echo "Downloading CUHK-Avenue dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mFu9sFG5p90urg6SmkVLwfpZjwIAfa32TinJLVv-11ygKXKXlDyA96nHpXWTsxT52m8RxlR5kFp03uU-_AmepLnmcLW4trLTu9IJejBuVahvNlbTbD7fA4fvq1fzdDL9s83uOX5JFKwN8P2e3X7TjNbQbKl0_HNU5FzIQy4QM31t-WDBMz29pfH2Ens5jXP0-bYDBVxUdNQnSSX5T9Dk_ew/avenue.tar.gz"
tar -xvf avenue.tar.gz
rm avenue.tar.gz

echo "Download CUHK-Avenue successfully..."
