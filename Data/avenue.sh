#!/bin/bash

echo "Downloading CUHK-Avenue dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mS8bbrfeD7Urmn0OWASYUcfrVLCTgcCwEBsTShdkWrrbfXTGLbKKMrT6KR94Nr9-DaFv1DBftJKqCzlCzG5phbgAPOy9V84BDtgzFceJpt0xwZstgPw_pZQR_E8jwmiw9QwhjMronyh2Yiy84huUbPEtFL6wt0TaN9KPQedwAMWaipj4w4di42BHwos5ESM5HZcim3Ng4xz5SPyN3btgrzg/avenue.tar.gz"
tar -xvf avenue.tar.gz
rm avenue.tar.gz

echo "Download CUHK-Avenue successfully..."