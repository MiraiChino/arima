today=`date '+%Y%m%d'`
echo $today
rm *.pickle
rm *.feather
rm models/*.pickle
rm models/*.index
rm racefeat/*.feather

echo python netkeiba.py
python netkeiba.py
echo python feature.py
python feature.py
echo python train.py
python train.py
echo python bakenhit.py
python bakenhit.py
echo docker compose -f docker-compose.deploy.yml build
docker compose -f docker-compose.deploy.yml build
echo docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
echo docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today