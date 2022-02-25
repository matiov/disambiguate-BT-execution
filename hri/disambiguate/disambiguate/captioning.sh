#!/bin/bash

grad_cam_path=$1
img_path=$2
target_dir=$3
expression=$4

cd $grad_cam_path
th captioning.lua -input_image_path $img_path -out_path $target_dir -caption "$expression" -gpuid 2