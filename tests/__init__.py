"""Unit test package for omicron."""
import json
import logging
import os
import subprocess
import sys
import asyncio
import socket
from contextlib import closing

import aiohttp
import cfg4py
import aioredis

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port

async def clear_cache(dsn):
    redis = await aioredis.create_redis(dsn)
    await redis.flushall()

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
    except Exception as e:
        logger.exception(e)
        return False


async def start_omega(timeout=60):
    port = find_free_port()

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
        stderr=subprocess.PIPE,
    )

    for i in range(timeout, 0, -1):
        await asyncio.sleep(1)
        if process.poll() is not None:
            # already exit
            out, err = process.communicate()
            logger.info("subprocess %s: %s", process.pid, out.decode("utf-8"))
            raise subprocess.SubprocessError(err.decode("utf-8"))
        if await is_local_omega_alive(port):
            logger.info("omega server is listen on %s", cfg.omega.urls.quotes_server)
            return process

    raise subprocess.SubprocessError("Omega server malfunction.")
