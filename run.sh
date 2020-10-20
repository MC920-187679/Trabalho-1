#!/usr/bin/bash

prog="python3 main.py $@"
build=build

# pastas de saída
for n in {1..11}
do
    mkdir -p $build/h$n
done
mkdir -p $build/h3h4

# convoluções
for file in $(ls imagens/*.png)
do
    out=$(basename $file)
    echo -n $out ...

    for n in {1..11}
    do
        $prog -o $build/h$n/$out $file h$n
    done
    $prog -o $build/h3h4/$out $file h3 h4

    echo ' 'done
done
