today=`date '+%Y%m%d'`
echo $today
python netkeiba.py
python feature_extractor.py
python train.py | tee train.log
docker compose -f docker-compose.deploy.yml build
docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today