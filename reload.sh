rm dist/*
poetry build
pip uninstall -y zillionare_omicron
pip install `ls dist/*.whl`
