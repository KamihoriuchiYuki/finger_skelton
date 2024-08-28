#!/bin/bash

SH_PATH=`dirname ${BASH_SOURCE:-0}`

#run 
alias rs="gnome-terminal --window -- bash $SH_PATH/bashes/realsense.bash"
alias sg="gnome-terminal --window -- bash $SH_PATH/bashes/sensor_glove.bash"
alias tr="gnome-terminal --window -- bash $SH_PATH/bashes/train.bash"
alias dh="gnome-terminal --window -- bash $SH_PATH/bashes/data_handler.bash"    
# alias tr="gnome-terminal --window -- bash -c 'bash $SH_PATH/bashes/train.bash; exec bash'"

#cd
alias cdls="cd $SH_PATH/../src/lstm_ae/src"

function sensors(){
    rs
    sleep 0.1s
    sg
    sleep 0.1s
    dh
}   