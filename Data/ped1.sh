#!/bin/bash

echo "Downloading UCSD-Ped1 dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mYp-s_RSs8flFeaThqT-Zwa_mSIgALD0g5M_ioQF20lFFnLYjYD8aaPR7pqVav4U_xxvRxMERy4Z0o_Fw0T8ysEHGwmaKuz135ajAwofSunZZpNV4e2E_IHW3mXwEy8-NMK7OF9U-Ntm1Pe9bxG-OH9acwL9Qg7EMa4vx-yGF_JRU3pTg-BPkIpuQaV8jhyAldwniIn-F1dGbiTLw08RZPg/ped1.tar.gz"
tar -xvf ped1.tar.gz
rm ped1.tar.gz

echo "Download UCSD-Ped1 successfully..."
