#!/bin/bash

echo "Downloading UCSD-Ped2 dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mFzDLdy1ZKsJawXtABkPGQsYhoZzeVYofrv5cKtvNS85CyUJcqwL0-P_PnzNvwrfEkIlQA9mQhld7CK9ohIa_lFvovPgNOZ3Z7BVnY-0sKA97Bv3OrnSU2Vkh9fl5ceDBo8PuCVoc_XHJN03Zj-v8q31cswu9RliBzujx_mLW4PxPi0cxui2j_n9xFp-S1Px_6H5a4_SGQBr_8EP8qsz3fA/ped2.tar.gz?download&psid=1"
mv "ped2.tar.gz?download&psid=1"    ped2.tar.gz
tar -xvf ped2.tar.gz
rm ped2.tar.gz

echo "Download UCSD-Ped2 successfully..."

echo "If encounters any ERRORS(I guess the download link in shell script is not permanent),
please manually download ped2.tar.gz from https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F
and run the following commands:

tar -xvf ped2.tar.gz
rm ped2.tar.gz

make sure the ped2 dataset is under the director of Data.
"
