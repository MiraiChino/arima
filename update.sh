today=`date '+%Y%m%d'`
echo $today
echo python netkeiba.py
python netkeiba.py
echo python feature_extractor.py
python feature_extractor.py
echo python train.py | tee temp/train_$today.log
python train.py | tee temp/train_$today.log
echo docker compose -f docker-compose.deploy.yml build
docker compose -f docker-compose.deploy.yml build
echo docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker tag lambda_arima us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
echo docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today
docker push us-west1-docker.pkg.dev/arima-339214/arima/lambda_arima:$today