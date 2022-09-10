
# 1. 安装

要使用Omicron来获取行情数据，请先安装[Omega](https://pypi.org/project/zillionare-omega/)，并按说明文档要求完成初始化配置。

然后在开发机上，运行下面的命令安装Omicron:

``` bash
    pip install zillionare-omicron
```

omicron依赖numpy, pandas, scipy, sklearn。这些库的体积比较大，因此在安装omicron时，请保持网络连接畅通，必要时，请添加阿里或者清华的PyPI镜像。

omicron还依赖于talib。如果在windows上安装出现困难时，请参考[这篇文章](https://blog.quantinsti.com/install-ta-lib-python)
