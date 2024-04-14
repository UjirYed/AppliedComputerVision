#!/bin/bash

input_file="output.txt"
total_lines=$(wc -l < "$input_file")
half_lines=$((total_lines / 2))

split -l "$half_lines" "$input_file" output
