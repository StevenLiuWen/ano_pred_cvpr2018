#!/bin/bash

echo "Downloading UCSD-Ped2 dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4mtXuyxi4Pb4MMZvlhLAJG34RmxFnpbCNPe9a9RK-Vl9QOlBWv3AEhXvGJVFb0RDFD_IOndEUNgRdlgTB5bGnJAVKEkYnm2CLSwCBD0WyB8UCzc-rHKg6XO6hQcggNTu4S1PhvYKeMuqHPlwKoa5tK8FxJbYWdP4ZGjeWTWeKS2z0qIlACqGYnq5K-VqUk5R5PcnqiTZaXQaBgKvteBjDXjA/ped2.tar.gz"
tar -xvf ped2.tar.gz
rm ped2.tar.gz

echo "Download UCSD-Ped2 successfully..."
