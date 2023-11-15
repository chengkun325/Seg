#!/bin/bash
set -e

# 压缩包名称
file="lightning_logs-$(date "+%Y%m%d-%H%M%S").zip"
# 把 lightning_logs 目录做成 zip 压缩包
zip -q -r "${file}" lightning_logs
# 通过 oss 上传到个人数据中的 backup 文件夹中
oss cp "${file}" oss://backup/
rm -f "${file}"
# 传输成功后关机
shutdown