#!/bin/bash
# if permission denied, run: chomd +x ros2_dev.sh

if [ -z $ros2_ws]; then
    export ros2_ws=~/Sensor-Glove
fi

source $ros2_ws/install/setup.bash
source $ros2_ws/exe/command.sh

function cbuild(){
    pushd $ros2_ws
    colcon build --symlink-install $*
    popd
}

function cbuild_pkg(){
	cbuild --packages-up-to $*
}

_cbuild_pkg() {
	local cur prev words cword split 
	_init_completion || return # この二行はセットで使うらしい  
	
	# 絞り込み後の配列を作成
	local comp_words_filtered=$(compgen -W '$(colcon list -n)' -- "$cur")
	COMPREPLY=( $comp_words_filtered )
}

complete -F _cbuild_pkg cbuild_pkg