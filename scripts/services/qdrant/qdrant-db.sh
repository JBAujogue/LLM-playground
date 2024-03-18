#!/usr/bin/env sh
name=qdrant-db
volume="$PWD/volumes/$name"

docker run \
    --name "$name" \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$volume":/qdrant/storage \
    qdrant/qdrant:latest