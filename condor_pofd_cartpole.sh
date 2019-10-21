#!/usr/bin/env bash
source /home/syseo/.bashrc
cd $2
/home/syseo/anaconda3/envs/sbl/bin/python userstudy_condor.py --process_id=$1