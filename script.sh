#!/bin/bash
containertag='ddp'
docker build . --tag $containertag ;
docker run --rm -v $(pwd):/ddp $containertag ;
