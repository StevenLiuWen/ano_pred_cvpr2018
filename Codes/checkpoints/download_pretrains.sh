#!/bin/bash

echo "Downloading trained models on ped1, ped2 and avenue datasets ....."

wget "https://ofhz9a.bn.files.1drv.com/y4mdc9K-lh3tfqBXiG-rSR04JARM20at-t2AtTo-7fUC-fadMB_x255o35v0J-YV4bnMW9m9XscVOKkITHI2bZLrgxZJJKXI4QEVDsi61KvsLQxI42elYWmm01F2kjI94onmbRrFYai7CkVNUspBgscY2vvEfd2c3qE2A_bcTW-Cp_6hBKpPEQClmwlT2QqTy-UwuzCmjyFfOrHqKeGkqtadQ/pretrains.tar.gz"
tar -xvf pretrains.tar.gz
rm pretrains.tar.gz

echo "Download pretrains successfully..."
