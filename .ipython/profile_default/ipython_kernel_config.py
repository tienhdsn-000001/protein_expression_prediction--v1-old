"""IPython configuration file."""

import logging
import requests

# pylint: disable=undefined-variable
c.IPKernelApp.extensions = ['bq_stats', 'google.cloud.bigquery', 'sql']

# We don't load the beatrix_jupyterlab extension for jupyterlab 4.x
enable_beatrix_jupyterlab = True
try:
  metadata_url = 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/enable-jupyterlab4-preview'
  headers = {'Metadata-Flavor': 'Google'}
  response = requests.get(metadata_url, headers=headers, timeout=1)
  response.raise_for_status()
  metadata_value = response.text
  if metadata_value == 'true':
    enable_beatrix_jupyterlab = False
except requests.exceptions.RequestException as e:
  logging.warning('Failed to fetch metadata: %s', e)
except Exception as e:  # pylint: disable=broad-except
  logging.warning('An unexpected error occurred: %s', e)

if enable_beatrix_jupyterlab:
  c.IPKernelApp.extensions.append('beatrix_jupyterlab')

c.InteractiveShellApp.matplotlib = 'inline'
