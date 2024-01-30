N_EPOCHS=4
N_CLIENTS=2  # needs to be at least 2, otherwise FL will hang
CLIENT_RESOURCES={"num_cpus": 4, "num_gpus": 0.0}
SERVER_IP="10.109.66.187:8080"  # ip of the server (run `ifconfig` on the compute node to get it)
COMMON_SERVER=False  # for split learning
USE_NUMPY_CLIENTS=False  # whether to use the numpy version of the clients or not
