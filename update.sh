today=`date '+%Y%m%d'`
echo $today

echo python netkeiba.py
python netkeiba.py

rm encoder*.pickle
rm feature*.pickle
rm ave*.feather
echo python feature.py
python feature.py

rm models/*.pickle
rm models/*.index
echo python train.py
python train.py

rm racefeat/*.feather
rm racefeat*.feather
rm breakpoint*.feather
rm bakenhit*.feather
echo python bakenhit.py
python bakenhit.py

echo docker compose -f docker-compose.deploy.yml build
docker compose -f docker-compose.deploy.yml build
echo docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
echo docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today