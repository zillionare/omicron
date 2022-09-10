
# 1. 安装

要使用Omicron来获取行情数据，请先安装[Omega](https://pypi.org/project/zillionare-omega/)，并按说明文档要求完成初始化配置。

然后在开发机上，运行下面的命令安装Omicron:

``` bash
    pip install zillionare-omicron
```

omicron依赖numpy, pandas, scipy, sklearn。这些库的体积比较大，因此在安装omicron时，请保持网络连接畅通，必要时，请添加阿里或者清华的PyPI镜像。

omicron还依赖于talib。omicron已经包含了ta-lib的python wrapper，但这个wrapper还依赖于ta-lib原生库，这部分需要您自行安装。

!!! 安装ta-lib
    === "Linux"
        请执行下面的脚本以完成安装：
        ```bash
        sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
        sudo apt-get install build-essential -y
        curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar -xzv -C /tmp/
        cd /tmp/ta-lib
        ./configure --prefix=/usr
        make
        sudo make install
        ```
    === "Windows"
        omicron仅在Ubuntu上进行过良好的测试。一般而言，Omicron也应该能够运行在64bit的windows上。但不推荐使用32bit windows。

        如果您是64位windows，请下载[ta-lib for 64bit windows](https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp38-cp38-win_amd64.whl)。然后通过 pip install {file_name} 来完成安装。
    === "MacOS"
        请通过`brew install ta-lib`来完成安装

    如果在安装中遇到任何问题，请参考[这篇文章](https://blog.quantinsti.com/install-ta-lib-python/)
