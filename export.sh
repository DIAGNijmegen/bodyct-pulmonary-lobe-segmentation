#!/usr/bin/env bash

./build.sh

docker save lobeseg | gzip -c > lobeseg.tar.gz
