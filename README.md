# Federated Domain Generalization Algorithm for Condition Monitoring (FDG-CM)

This repository presents the algorithm developed in the FDG_CM paper (previously named FTL-TP). This algorithm is a novel Federated Domain Generalization (FDG) technique designed specifically for condition monitoring in manufacturing processes. This framework has been tested on Ultrasonic metal welding data but can be extensible on other condition monitoring data. 

**Note:** The code is currently under development and is not yet complete.

## Paper Information


The paper associated with this repository has been accepted to the *Journal of Manufacturing Systems (JMS)* and is currently under review. Once finalized, it will provide a comprehensive overview of the FDG_CM algorithm and its applications.

## Repository Structure

- **src/**: Contains all client and server code.
- **run.py**: Script to execute all client codes. You can modify this script to select specific clients for execution on each edge device.

## Requirements

To run the FDG_CM algorithm, you will need:

- One CPU core per client node.
- For example, to run 16 clients, you will require 4 Raspberry Pis, each equipped with 4 CPU cores.
- Using RabbitMQ broker for routing messages.

## Instructions

1. Navigate to the `src/` folder.
2. Modify the Clients to use the data you want.
3. Use the `run.py` script to execute the desired clients.

Feel free to modify `run.py` to fit your specific deployment needs.

### Author Links
Link to the Arxiv paper: 
https://arxiv.org/abs/2404.13278
