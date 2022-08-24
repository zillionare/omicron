#!/bin/bash

# !!!NOTICE!!
# Personal token with full access rights is required to run this scripts
# Once you got persona token, set environment variable GH_TOKEN with it

# uncomment the following to create repo and push code to github
# gh repo create omicron --public
# git remote add origin git@github.com:zillionare/omicron.git
# git add .
# pre-commit run --all-files
# git add .
# git commit -m "Initial commit by ppw"
# git branch -M main

# Uncomment the following to config github secret used by github workflow.
gh secret set PERSONAL_TOKEN --body $GH_TOKEN
gh secret set PYPI_API_TOKEN --body $PYPI_API_TOKEN
gh secret set TEST_PYPI_API_TOKEN --body $TEST_PYPI_API_TOKEN

# uncomment the following if you need to setup email notification
gh secret set BUILD_NOTIFY_MAIL_SERVER --body $BUILD_NOTIFY_MAIL_SERVER
gh secret set BUILD_NOTIFY_MAIL_PORT --body $BUILD_NOTIFY_MAIL_PORT
gh secret set BUILD_NOTIFY_MAIL_FROM --body $BUILD_NOTIFY_MAIL_FROM
gh secret set BUILD_NOTIFY_MAIL_PASSWORD --body $BUILD_NOTIFY_MAIL_PASSWORD
gh secret set BUILD_NOTIFY_MAIL_RCPT --body $BUILD_NOTIFY_MAIL_RCPT

# git push -u origin main
