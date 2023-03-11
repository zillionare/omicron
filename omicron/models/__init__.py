#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import cfg4py

from omicron.dal.influx.influxclient import InfluxClient

logger = logging.getLogger(__name__)


def get_influx_client():
    cfg = cfg4py.get_instance()
    url = cfg.influxdb.url
    token = cfg.influxdb.token
    bucket_name = cfg.influxdb.bucket_name
    org = cfg.influxdb.org
    compressed = cfg.influxdb.enable_compress
    return InfluxClient(url, token, bucket_name, org=org, enable_compress=compressed)
