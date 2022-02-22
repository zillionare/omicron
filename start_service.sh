if [ $IS_GITHUB ]; then
    echo "runs on github, skipping start_service.sh"
    exit 0
fi

echo "本地测试环境，将初始化redis, postgres, minio和influxdb!"

export TZ=Asia/Shanghai
sudo -E apt-get update

echo "初始化redis容器"
sudo docker run -d --name tox-redis -p 6379:6379 redis

echo "初始化postgres容器"

sudo docker run --name tox-postgres -p 5432:5432 -e POSTGRES_PASSWORD=123456 -e POSTGRES_USER=zillionare -e POSTGRES_DB=zillionare -d postgres

sudo -E apt-get install --yes --no-install-recommends postgresql-client

sleep 3

PGPASSWORD=123456 psql -U zillionare -h localhost --dbname=zillionare --file=omicron/config/sql/init.sql
PGPASSWORD=123456 psql -U zillionare -h localhost --dbname=zillionare --file=omicron/config/sql/v1.0.sql

echo "初始化influxdb容器"
sudo docker run -d -p 8086:8086 --name tox-influxdb influxdb
sleep 5
sudo docker exec -i tox-influxdb bash -c 'influx setup --username my-user --password my-password --org my-org --bucket my-bucket --token my-token --force'

echo "初始化minio容器"
sudo docker run -d -p 9000:9000 -p 9001:9001 --name tox-minio minio/minio server /tmp --console-address ":9001"
