=====
安装
=====

要使用Omicron来获取行情数据，请先安装 `Omega <https://pypi.org/project/zillionare-omega/>`_，并按说明文档要求完成初始化配置。

然后在开发机上，运行下面的命令安装Omicron:

.. code:: console

    pip install zillionare-omicron

同样地，我们推荐您为自己的项目创建虚拟环境。如果您是新建工程的话，我们推荐使用Cookiecutter来创建工程框架，并参照 `大富翁开发规范`_ 来完成项目的配置。


.. hint::

    对于Omicron的协作开发者，需要注意以下事项：

    1. 您必须遵守 `大富翁开发规范`_ 来进行开发。

    2. Omicron在测试和运行时依赖于对应版本的Omega。在发行版的Omicron中，并不包含此依赖。在开发过程中，为了单元测试需要，您可能需要安装对应版本的Omega。

        ```text
        pip install zillionare-omega
        ```
    3. 如果您增加的功能同时需要更改Omega，则应该使用editable installation：

        ```text
        pip install -e %path_to_omega%
        ```
    4. 开发完成，更新两个Project中对应依赖的版本号。


 .. _`大富翁开发规范`: https://zillionare.readthedocs.io/zh_CN/latest/developer_guide.html
