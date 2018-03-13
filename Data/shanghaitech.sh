#!/bin/bash
echo "download ShanghaiTech-Campus dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mb9ehfPcEc2QJOWHHdhumdqt5jxTobXbtJQY6Q-5HWfoTWSSpDdFEN7Mkx861al_K9Tt3zQdzEIS7i3gRUVi0lfOsKrLaKV2sV6qHaTKmWPMUvOfMhMh2lScycBEmp4NasUkKJR2eftgbZ5XzHui03_LVL875RK1Z5sTtPADPD2TuPwHG8_hGhtfQOtJUWbqMvh0XmGZq-qYP9YIQKbe9Lw/shanghaitech.tar.gz"
tar -xvf shanghaitech.tar.gz
rm shanghaitech.tar.gz

echo "download ShanghaiTech-Campus successfully..."
