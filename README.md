# Implementation of [Flanc](https://openreview.net/pdf?id=wfel7CjOYk) using MPI

## Introduction

This is an implementation of [Resource-Adaptive Federated Learning with All-In-One Neural Composition](https://openreview.net/pdf?id=wfel7CjOYk) using MPI and Pytorch.

The official implementation of this paper is [here](https://github.com/HarukiYqM/All-In-One-Neural-Composition).

* Global hyperparameters are defined in config.yml
* server_main.py is the main file to run server code
* client_main.py is the main file to run client code

## Start program

If you want to run Flanc with 10 clients, you can input this command in the console:

``
mpiexec --oversubscribe -n 1 python server_main.py : -n 10 python client_main.py
``

Each client and the server runs as a process, which communicates with each other by MPI.

Make sure that the value of hyperparameter "active_client_num" in config.yml equals the number of clients.
