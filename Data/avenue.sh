#!/bin/bash

echo "Downloading CUHK-Avenue dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4m4fpDJfxvClUUg4yfbH22DpWmnN8smMTSoK0tPyEB2VUQmsD0oUkURguUYhQABYcDkdXvpseAe2G4gxjdnssPWERMbyGA8z6tk-pU6V4fNvRjZBdH3P6joeAEbOPXcK0ZhQCRqDVROdbZQ0vMZjoXiRf2Kvs_o175MW1xLKvfOmIMcw3ZhtF6iOmvIvMfmP8RcZJNbp8CSOwySQgONpkODQ/avenue.tar.gz?download&psid=1"
mv "avenue.tar.gz?download&psid=1" avenue.tar.gz
tar -xvf avenue.tar.gz
rm avenue.tar.gz

echo "Download CUHK-Avenue successfully..."

echo "If encounters any ERRORS(I guess the download link in shell script is not permanent),
please manually download avenue.tar.gz from https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F
and run the following commands:

tar -xvf avenue.tar.gz
rm avenue.tar.gz

make sure the avenue dataset is under the director of Data.
"
