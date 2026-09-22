from urllib.request import urlopen
import os

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--test_input', action='store', type=str, required=True, dest='test_input')


args = arg_parser.parse_args()
print(args)

id = args.id

test_input = args.test_input.replace('"','')



print("Input ricevuto:", test_input)

for name in (
    "OAUTH_ACCESS_TOKEN",
    "OAUTH_REFRESH_TOKEN",
    "JUPYTERHUB_API_TOKEN",
    "JUPYTERHUB_USER",
):
    print(f"{name}: {'presente' if os.getenv(name) else 'assente'}")


url = "http://sftpgo-gateway.naavre.svc.cluster.local/health"

with urlopen(url, timeout=10) as response:
    print("Gateway:", response.status, response.read().decode())

