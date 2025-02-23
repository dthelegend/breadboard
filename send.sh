#!/bin/sh

screen -S arduino -X readreg p a.out &&
sleep 0.5 &&
screen -S arduino -X paste p &&
echo "done"
