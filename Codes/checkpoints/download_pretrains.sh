#!/bin/bash

echo "Downloading trained models on ped1, ped2 and avenue datasets ....."

wget "https://ofhz9a.bn.files.1drv.com/y4m5lC_SnkDiTcKjKEiue7uKKHX_jM7LojjlsjpurNHC8gkOj0MjgqdKrj6YJwLNFMAb649j07rheaBeS-B8JmYwGc3wy6Zb7T0ICYBzz9PdheTGxHWGsLCxJ7MpaA4Rj6V0KmtAyoUYbdeNQVWEAPAZtVn1ikrdslLVVvKB1doyWRaTnIKjCiXIybbXG-6VtZ4uw10H_PrBFTEq6cBeqr2CQ/pretrains.tar.gz?download&psid=1"
mv "pretrains.tar.gz?download&psid=1" pretrains.tar.gz
tar -xvf pretrains.tar.gz
rm pretrains.tar.gz

echo "Download pretrains successfully..."

echo "If encounters any ERRORS(I guess the download link in shell script is not permanent),
please manually download pretrains.tar.gz from https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F
and run the following commands:

tar -xvf pretrains.tar.gz
rm pretrains.tar.gz

make sure the pretrains is under the director of Codes/checkpoints.
"