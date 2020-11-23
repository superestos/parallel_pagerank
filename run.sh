#!/bin/bash
if [ $1 == 'stanford' ]
then
    OPTION='-f datasets/stanford.bin -n 281903 -e 2312497'
elif [ $1 == 'livejournal' ]
then
    OPTION='-f datasets/livejournal.bin -n 4847571 -e 68993773'
elif [ $1 == 'google' ]
then
    OPTION='-f datasets/google.bin -n 875713 -e 5105039'
fi

./pagerank $OPTION