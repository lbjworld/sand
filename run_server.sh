#!/bin/bash

docker run --rm -it -v `pwd`/:/code --workdir /code -p "8000:8888" ml:latest
