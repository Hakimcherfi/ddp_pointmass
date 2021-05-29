#!/bin/bash
containertag='ddp'
docker build . --tag $containertag ;
docker run -v $(pwd):/ddp $containertag ;
