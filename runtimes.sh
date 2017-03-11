# test different epoch runtimes 

epochs="50 100 150 200 250 300"
for i in $pops; do
    echo "$i: " >> $2
    { time python $1 ~/Pictures/goog.png ./results -epochs $i -p 100 ;} 2>> $2

done 

