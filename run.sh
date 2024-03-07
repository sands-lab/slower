#!/bin/bash

cleanup() {
    echo "Stopping background process"
    kill "$bg_pid"
}


python usage/mnist/grpc/run_server.py &
trap cleanup SIGINT
bg_pid=$!


sleep 10

python usage/mnist/grpc/run_client.py

wait "$bg_pid"

cleanup
