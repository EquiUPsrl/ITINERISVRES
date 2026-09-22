from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen
from urllib.parse import urlencode
import os

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--test_input', action='store', type=str, required=True, dest='test_input')

arg_parser.add_argument('--param_email', action='store', type=str, required=True, dest='param_email')

args = arg_parser.parse_args()
print(args)

id = args.id

test_input = args.test_input.replace('"','')

param_email = args.param_email.replace('"','')


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




if not param_email:
    raise ValueError("Specifica param_email all'avvio del workflow")

source_file = Path("/tmp/data/test_gateway_workflow.txt")
remote_path = "output/test_gateway_workflow.txt"

url = (
    "http://sftpgo-gateway.naavre.svc.cluster.local/upload?"
    + urlencode({"email": param_email, "path": remote_path})
)

request = Request(
    url,
    data=source_file.read_bytes(),
    headers={"Content-Type": "application/octet-stream"},
    method="POST",
)

with urlopen(request, timeout=60) as response:
    print("Upload:", response.status, response.read().decode())

