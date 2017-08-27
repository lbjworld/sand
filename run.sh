#!/bin/bash

docker run --rm -it -v `pwd`/:/code --workdir /code ml:latest bash
