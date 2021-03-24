#!/usr/bin/env python
# -*- coding: utf-8 -*-


class QuotesServerConnectionError(BaseException):
    pass


class FetcherQuotaError(BaseException):
    """quotes fetcher quota exceed"""

    pass
