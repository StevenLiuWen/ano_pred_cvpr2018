#!/bin/bash

echo "Downloading trained models on ped1, ped2 and avenue datasets ....."

wget "https://ofhz9a.bn.files.1drv.com/y4mHfGdUxGoa7NnnI-eIlTqInymvmHyDOSGGw5zKM08jOGukHKdYdxmtZiEEh-rCAWK7oTDTstQ5bKazvjdyTtsIUW7zxcKnVgIsgZg6DpEb-Qdq83Zmnnw6nv7pX5HhiOkMxc42CLl65QK0A2Mv1Cmj-062Pyodm-Mt5r24Id3_glS0NT6BdvAp7-VbevkXygnmXQrcXRQU6d0y1cHlZJ2ig/pretrains.tar.gz"
tar -xvf pretrains.tar.gz
rm pretrains.tar.gz

echo "Download pretrains successfully..."


