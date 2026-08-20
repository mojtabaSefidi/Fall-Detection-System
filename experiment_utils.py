from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from importlib import metadata
import json
import logging
import platform
import shutil
import sys
import time


PACKAGE_NAMES = [
    'matplotlib', 'numpy', 'pandas', 'scikit-learn', 'seaborn',
    'tensorflow', 'tqdm',
]


def installed_package_versions():
  versions = {}
  for package_name in PACKAGE_NAMES:
    try:
      versions[package_name] = metadata.version(package_name)
    except metadata.PackageNotFoundError:
      versions[package_name] = None
  return versions


def configure_logging(level=logging.INFO):
  logger = logging.getLogger('fall_detection')
  if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'))
    logger.addHandler(handler)
  logger.setLevel(level)
  logger.propagate = False
  return logger


def stage_directory(source, destination, logger=None):
  """Optionally copy a dataset directory to faster local storage."""
  source = Path(source)
  destination = Path(destination)
  if not source.exists():
    raise FileNotFoundError(f'Staging source does not exist: {source}')
  if source.resolve() == destination.resolve():
    return source
  logger = logger or configure_logging()
  if destination.exists():
    logger.info('Using existing local dataset stage: %s', destination)
    return destination
  logger.info('Staging %s to %s', source, destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(source, destination)
  logger.info('Dataset staging completed: %s', destination)
  return destination


class RunTracker:

  def __init__(self, logger=None):
    self.logger = logger or configure_logging()
    self.started_at = datetime.now(timezone.utc).isoformat()
    self.timings = {}

  @contextmanager
  def stage(self, name):
    self.logger.info('%s started', name)
    started = time.perf_counter()
    try:
      yield
    finally:
      duration = time.perf_counter() - started
      self.timings[name] = duration
      self.logger.info('%s finished in %.2f seconds', name, duration)

  def save_manifest(self, path, config=None, dataset=None, extra=None):
    manifest = {
        'started_at_utc': self.started_at,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'package_versions': installed_package_versions(),
        'timings_seconds': self.timings,
        'config': config or {},
        'dataset': dataset or {},
        'extra': extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as manifest_file:
      json.dump(manifest, manifest_file, indent=2, default=str)
    self.logger.info('Run manifest saved to %s', path)
    return manifest
