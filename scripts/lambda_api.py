import json
import os
import click
import requests
import time
import yaml


LAMBDA_API_ROOT = "https://cloud.lambda.ai/api/v1"


class LambdaAPI:

  def __init__(self, api_key=None):
    self._api_root = LAMBDA_API_ROOT
    self._api_key = api_key or os.getenv("LAMBDA_API_KEY")
    if self._api_key is None:
      raise ValueError("No environment variable found for LAMBDA_API_KEY")

  def get(self, endpoint, params=None):
    url = f"{self._api_root}/{endpoint}"
    headers = {
      "Authorization": f"Bearer {self._api_key}"
    }
    response = requests.get(url, headers=headers, params=params)
    return response
  
  def post(self, endpoint, data=None):
    url = f"{self._api_root}/{endpoint}"
    headers = {
      "Authorization": f"Bearer {self._api_key}"
    }
    response = requests.post(url, headers=headers, json=data)
    return response

  def list_instances(self):
    return self.get("instances")
  
  def list_instance_types(self, region=None):
    response = self.get("instance-types")
    if region is None:
      return response.json()
    
    results = {}
    for type_name, type_data in response.json()["data"].items():
      regions = [r["name"] for r in type_data.get("regions_with_capacity_available", [])]
      if region in regions:
        results[type_name] = type_data
    return {"data": results}

  def create_instance(self, spec_filepath):
    with open(spec_filepath, "r") as f:
      spec = yaml.safe_load(f)
    print(spec)
    return self.post("instance-operations/launch", data=spec)
  
  def terminate_instance(self, instance_id=None):
    if instance_id is None:
      response = self.list_instances()
      instance_data = response.json()["data"]
      if not instance_data:
        raise ValueError("No instances are running")
      instance_id = instance_data[0].get("id")

    if click.confirm(f"About to delete instance {instance_id}?", default=False):
      post_data = {
        "instance_ids": [ instance_id ]
      }
      response = self.post("instance-operations/terminate", data=post_data)
      return response


@click.group()
def cli():
  pass


@cli.command(
    help="Launch an instance. Takes one argument, the filename of the instance config as yaml.")
@click.argument("config")
@click.option("--retry_seconds", "-r", default=0)
def launch_instance(config, retry_seconds):
  api = LambdaAPI()

  def _try_launch():
    response = api.create_instance(config).json()
    if ("error" in response and 
        response["error"].get("code") == "instance-operations/launch/insufficient-capacity"):
      return (False, response)
    return (True, response)

  success = False
  while not success:
    success, response = _try_launch()
    print(success, response) # FIXME
    if not success:
      print(f"No capacity. Retrying in {retry_seconds} seconds")
      time.sleep(retry_seconds)
  print(response)


@cli.command()
def list_instances():
  api = LambdaAPI()
  response = api.list_instances()
  print(response.text)


@cli.command()
@click.argument("region", default="us-east-1")
@click.option("-v", "--verbose", is_flag=True)
def list_instance_types(region, verbose):
  api = LambdaAPI()
  response = api.list_instance_types(region)
  instance_types = sorted(
    [v["instance_type"] for v in response["data"].values()],
    key=lambda x: x["price_cents_per_hour"]
  )
  if verbose:
    print(json.dumps(instance_types, indent=2))
  else:
    info_strs = [
      f"{info['name']} (${info['price_cents_per_hour']/100})"
      for info in instance_types
    ]
    print("\n".join(info_strs))


@cli.command()
@click.argument("instance_id", required=False)
def terminate_instance(instance_id):
  api = LambdaAPI()
  api.terminate_instance(instance_id=instance_id)


if __name__ == "__main__":
  cli()
