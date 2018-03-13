#!/bin/bash
echo "download ShanghaiTech-Campus dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mavKKJgKjjUPr3CnqA6u-xYxU3DIwYPAhAVv5UhQpf82uT31Ueljk3qxkPlcwlCuc0oSLhb5RfDN_vJKv3qvyOAoKP1NFNq3A6xiAtcYR0F2Xm2AXxWEabD-yPR49bwHMGWKPKItSiw_bPhvrretOmPf9QqxEoc7TqrN0A8ZGHwl5ASdtT2n2e3TwZyMIRQfSCQ0yfDZKfml7WM2so9G6Nw/shanghaitech.tar.gz"
tar -xvf shanghaitech.tar.gz
rm shanghaitech.tar.gz

echo "download ShanghaiTech-Campus successfully..."
