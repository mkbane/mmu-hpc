#!/bin/bash


t1=`head -1 $1|awk '{print $2}'`;echo t1 $t1
gnuplot -persist <<EOF
plot '$1' w lp t '$1 times', '$2' w lp t '$2 times'
EOF

# gnuplot -persist <<EOF
# set logscale xy
# plot '$1' w lp t 'log-log time'
# EOF

# gnuplot -persist <<EOF
# plot [1:130][1:130] '$1' u 1:($t1/\$2) w lp t 'speed-up, Sp'
# EOF

# gnuplot -persist <<EOF
# plot [][0:] '$1' u 1:(100*$t1/\$2/\$1) w lp t 'efficiency, Ep'
# EOF
