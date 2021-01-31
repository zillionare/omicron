"""Unit test package for omicron."""
import json
import logging
import os
import subprocess
import sys
import time

import aiohttp
import cfg4py

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


def init_test_env():
    os.environ[cfg4py.envar] = "DEV"
    src_dir = os.path.dirname(__file__)
    config_path = os.path.join(src_dir, "../omicron/config")

    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)-1.1s %(name)s:%(funcName)s:%(lineno)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)

    return cfg4py.init(config_path, False)


async def is_local_omega_alive(port: int = 3181):
    try:
        url = f"http://localhost:{port}/sys/version"
        async with aiohttp.ClientSession() as client:
            async with client.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        return True
    except Exception:
        return False


async def start_omega(port: int = 3181):
    if await is_local_omega_alive(port):
        return None

    cfg.omega.urls.quotes_server = f"http://localhost:{port}"
    account = os.environ["JQ_ACCOUNT"]
    password = os.environ["JQ_PASSWORD"]

    # hack: by default postgres is disabled, but we need it enabled for ut
    cfg_ = json.dumps({"postgres": {"dsn": cfg.postgres.dsn, "enabled": "true"}})

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omega.app",
            "start",
            "--impl=jqadaptor",
            f"--cfg={cfg_}",
            f"--account={account}",
            f"--password={password}",
            f"--port={port}",
        ],
        env=os.environ,
        stdout=subprocess.PIPE,
    )

    for i in range(20, 0, -1):
        if process.poll() is not None:
            # already exit
            msg = f"Omega server exited abnormally with status {process.returncode}"
            raise subprocess.SubprocessError(msg)
        if await is_local_omega_alive():
            return process

        time.sleep(10)
    raise subprocess.SubprocessError("Omega server malfunction.")
