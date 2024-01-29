from slower.client.app import start_client
from usage.mnist.sl import MnistClient

address = "10.109.66.187:8080"
start_client(server_address=address, client_fn= MnistClient)
