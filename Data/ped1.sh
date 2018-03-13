#!/bin/bash

echo "Downloading UCSD-Ped1 dataset....."

wget "https://ofhz9a.bn.files.1drv.com/y4m5ZrdN62Hy303ATx2p6Cogia4Ewpwnye8HgJ7qgWdFJ6gaNKGErugal2lfpdr2h65rjArbhrID9mxfSIf2WfXvh9AJf40xwcEWxEAuTp_-gSkfyLAt4Ef7xkJko4InRzUJz-3bdvV77dmBuYSl9LljLyP6908E4EyvPEkMI3pHrNP5QmiJSsHN6jFwtOHpZuUG8UeJGpqb-TwKjNxFrEpjQ/ped1.tar.gz"
tar -xvf ped1.tar.gz
rm ped1.tar.gz

echo "Download UCSD-Ped1 successfully..."
