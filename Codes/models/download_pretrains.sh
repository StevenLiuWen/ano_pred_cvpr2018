#!/bin/bash

echo "Downloading trained models on ped1, ped2 and avenue datasets ....."

wget "https://ofhz9a.bn.files.1drv.com/y4mljH6-fb1fLS6B9cJg5kst-wanKhmgoA8mWk_8y8atGlL4SQqEqZRljA8LuLkxkvm-36PArbAsx-uYKxzfUX1s8otO8h0Rv7tag9Agx1h8_RsTPhQo-aft0QSybwpvcvOkegqfLVg2UJjTzIsxgizIYCVDIpunmpasyB3t4mchRGZetZQGiKP4fS2Vfb4c3tj1jxzpN61Pmou9QaB3VAG0w/pretrains.tar.gz"
tar -xvf pretrains.tar.gz
rm pretrains.tar.gz

echo "Download pretrains successfully..."


