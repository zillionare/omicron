#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a awesome
        python script!"""
import logging

logger = logging.getLogger(__file__)


class Events:
    SECURITY_LIST_UPDATED = "quotes/security_list_updated"
    OMEGA_WORKER_JOIN = "omega/worker_join"
    OMEGA_WORKER_LEAVE = "omega/worker_leave"
    OMEGA_APP_START = "omega/app_start"
