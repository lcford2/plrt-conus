#!/bin/bash

# find ssh client
ssh_sessions="$(sudo lsof -i TCP -s tcp:established -n | grep 'ssh->' | grep $USER)"
ip="$(echo $ssh_sessions | grep -oP '>(\d{1,3}\.){3}\d{1,3}')"

ip_array=($ip)
ip="${ip_array[0]}"

# trim off arrow
ip=${ip:1}

if [ $# -eq 0 ]; then
    username="lucas"
else
    username=$1
fi

remotehost="$username@$ip"
rsync directories
echo "RSYNCing with $remotehost"
rsync -azP ../data/ $remotehost:~/projects/plrt-conus/data
rsync -azP ../results/ $remotehost:~/projects/plrt-conus/results
