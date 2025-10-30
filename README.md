# 一锤定音大模型工具（锤子代码）

## 使用

脚本主要用于在Linux（Ubuntu）环境中使用，快速实现LLM,VLM等大模型的推理、微调、合并等。

特性如下：

1. 支持多个镜像下载模型文件，支持全部下载/单文件、多文件下载；
2. 支持微调参数设置（提供默认参数、默认数据集）；
3. 自动生成执行脚本。

## 目前限制

1. 需要手动升级ms-swift工具才能适配最新大模型；
2. [Model Type](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html) 仅支持手动录入；
3. VLM的支持不如LLM。

## 感谢

感谢ModelScope团队的付出和努力！
