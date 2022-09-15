
# 1. 安装

要使用Omicron来获取行情数据，请先安装[Omega](https://pypi.org/project/zillionare-omega/)，并按说明文档要求完成初始化配置。

然后在开发机上，运行下面的命令安装Omicron:

``` bash
    pip install zillionare-omicron
```

omicron依赖numpy, pandas, scipy, sklearn。这些库的体积比较大，因此在安装omicron时，请保持网络连接畅通，必要时，请添加阿里或者清华的PyPI镜像。

omicron还依赖于talib, zigzag, ciso8601等高性能的C/C++库。安装这些库往往需要在您本机执行一个编译过程。请遵循以下步骤完成：

!!! 安装原生库
    === "Windows"
        **注意我们不支持32位windows**

        请跟随[windows下安装omicron](_static/Omicron_Windows10.docx)来完成安装。
    === "Linux"
        1. 请执行下面的脚本以完成ta-lib的安装
        ```bash
        sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
        sudo apt-get install build-essential -y
        curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar -xzv -C /tmp/
        cd /tmp/ta-lib
        ./configure --prefix=/usr
        make
        sudo make install
        ```
        1. 现在安装omicron，所有其它依赖的安装将自动完成。

    === "MacOS"
        1. 请通过`brew install ta-lib`来完成ta-lib的安装
        2. 现在安装omicron，所有其它依赖的安装都将自动完成。

# 2. 常见问题
## 无法访问aka.ms
如果遇到aka.ms无法访问的问题，有可能是IP地址解析的问题。请以管理员权限，打开并编辑位于c:\windows\system32\drivers\etc\下的hosts文件，将此行加入到文件中：
```
23.41.86.106 aka.ms
```
![](https://images.jieyu.ai/images/202209/20220915185255.png)
