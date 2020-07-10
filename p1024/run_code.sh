#!/bin/bash

module load cuda
make
./massif.x
