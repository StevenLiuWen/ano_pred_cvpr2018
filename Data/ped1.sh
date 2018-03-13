#!/bin/bash

echo "Downloading UCSD-Ped1 dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mP5HrUYe3m0KnhIA3KbOaqlFEKpvCmqepz-C9UDoIUgO4i0WuW9Dm-J-98qYXivCdniC-_mYHq9r4t25im6XogBz-INqqktYE2Rc38vkKKwM1iFZ_uWxoGon4QniumO2gNLscP9N9wNw6fWD8GqIYqOUVe_UO9svbF0RpeRpAbSe82uHJ9qqmN2q-mZ9prbrScwsolPEv_IxprXqgjG5Plw/ped1.tar.gz?download&psid=1"
mv "ped1.tar.gz?download&psid=1" ped1.tar.gz
tar -xvf ped1.tar.gz
rm ped1.tar.gz

echo "Download UCSD-Ped1 successfully..."

echo "If encounters any ERRORS(I guess the download link in shell script is not permanent),
please manually download ped1.tar.gz from https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F
and run the following commands:

tar -xvf ped1.tar.gz
rm ped1.tar.gz

make sure the ped1 dataset is under the director of Data.
"

