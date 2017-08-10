#!/bin/bash

docker run --rm -it -v `pwd`/:/code --workdir /code gw000/keras:2.0.6-py2-tf-cpu bash
