from slower.client.app import start_client

from usage.yelp_review.raw.yelp_raw_client import YelpRawClient
import usage.yelp_review.constants as constants


def main():
    start_client(server_address=constants.SERVER_IP, client_fn=YelpRawClient)


if __name__ == "__main__":
    main()
