#!/bin/bash
echo "download ShanghaiTech-Campus dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mZ-bxF_FckWxjvJKGdcIkCr4PZOK3JQIbVqcv_1IE8QnAvQzlCnIqAWiWI6l_NVpBcfizN_6EOYc01NMPCiEj_YCFOyBVK1ZjczoTHClYXry87x5DrzcimwVXttkPtHBytzj43XKWFoOIFyZqpJDUL5o5GoZnfp5g3i1tthSsuIy4YnMMOup1tebJ8jb_Kqb09kksykw2YE-C-0pD5ovsVQ/shanghaitech.tar.gz?download&psid=1"
mv "shanghaitech.tar.gz?download&psid=1"    shanghaitech.tar.gz
tar -xvf shanghaitech.tar.gz
rm shanghaitech.tar.gz

echo "download ShanghaiTech-Campus successfully..."

echo "If encounters any ERRORS(I guess the download link in shell script is not permanent),
please manually download shanghaitech.tar.gz from https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F
and run the following commands:

tar -xvf shanghaitech.tar.gz
rm shanghaitech.tar.gz

make sure the shanghaitech dataset is under the director of Data.
"