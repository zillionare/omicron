if [ $IS_GITHUB ]; then
    echo "runs on github, skipping start_service.sh"
    exit 0
fi

echo "本地测试环境，将初始化redis和haystore!"

export TZ=Asia/Shanghai
# sudo -E apt-get update

echo "初始化网络"

echo "初始化redis容器"
sudo docker run -d --name tox-redis --network host redis

echo "初始化haystore容器"
sudo docker run -d --name tox-haystore --network host --ulimit nofile=262144:262144 -e CLICKHOUSE_DB=tests -e CLICKHOUSE_USER=default -e CLICKHOUSE_PASSWORD=123456 -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 clickhouse/clickhouse-server
