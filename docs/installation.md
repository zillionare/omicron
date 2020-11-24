# 安装指南

Omicron是Zillionare的公共核心库。它应该随着其它服务（比如Zillionare-omega）一
起安装和配置，您通常无须特别进行任何操作。

对于Omicron的协作开发者，需要注意以下事项：

1. Omicron在测试和运行时依赖于对应版本的Omega。在发行版的Omicron中，并不包含此依赖。在开发过程中，为了单元测试需要，您可能需要安装对应版本的Omega。

    ```text
    pip install zillionare-omega
    ```
2. 如果您增加的功能同时需要更改Omega，则应该使用editable installation：

    ```text
    pip install -e %path_to_omega%
    ```
3. 开发完成，更新两个Project中对应依赖的版本号。哪样？就这样！