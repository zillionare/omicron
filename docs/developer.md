## Omicron的开发流程
Omicron遵循[ppw](https://zillionare.github.io/python-project-wizard)定义的开发流程和代码规范。您可以阅读[tutorial](https://zillionare.github.io/python-project-wizard/tutorial/)来了解更多。

简单来说，通过ppw构建的工程，具有以下能力：
### 基于poetry进行依赖管理
1. 通过poetry add给项目增加新的依赖。如果依赖项仅在开发环境下使用，请增加为Extra项，并正确归类为dev, doc和test中的一类。
2. 使用poetry lock来锁定依赖的版本。
3. 使用poetry update更新依赖项。
### flake8, isort, black
omicron使用flake8, isort和black进行语法检查和代码格式化
### pre-commit
使用pre-commit来确保提交的代码都符合规范。如果是刚下载代码，请运行pre-commit install安装钩子。
## TODO: 将通用部分转换到大富翁的开发者指南中
## 如何进行单元测试？
### 设置环境变量
Omicron在notify包中提供了发送邮件和钉钉消息的功能。在进行单元测试前，需要设置相关的环境变量：

```bash
DINGTALK_ACCESS_TOKEN=?
DINGTALK_SECRET=?

export MAIL_FROM=?
export MAIL_SERVER=?
export MAIL_TO=?
export MAIL_PASSWORD=?
```

上述环境变量已在gh://zillionare/omicron中设置。如果您fork了omicron并且想通过github actions进行测试，请在您的repo中设置相应的secrets。

### 启动测试
通过tox来运行测试。tox将启动必要的测试环境（通过`stop_service.sh`和`start_service.sh`）。

### 文档
文档由两部分组成。一部分是项目文档，存放在docs目录下。另一部分是API文档，它们从源代码的注释中提取。生成文档的工具是mkdocs。API文档的提取则由mkdocs的插件mkdocstrings提取。
