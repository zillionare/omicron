pip uninstall -y zillionare-omicron
rm dist/*
poetry build
pip install -qq dist/zillionare_omicron-1.1.1-py3-none-any.whl
