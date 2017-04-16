# mgr-tensorflow
Tensorflow python API

## Development
```
export FLASK_APP=flaskapp.py
export FLASK_DEBUG=1
flask run
```

## Deployment
```
git push heroku master
```

## Server install
http://www.datasciencebytes.com/bytes/2015/02/24/running-a-flask-app-on-aws-ec2/
```
cd flaskapp/
git pull origin master
sudo pip install -r requirements.txt
sudo apachectl restart
```

Test tensorflow version
```
python -c 'import tensorflow as tf; print(tf.__version__)'
```

## Server logs (linux)
#### ERROR_LOGS
```
tail -f /var/log/apache2/error.log
```

#### APP_LOGS
```
tail -f /var/log/apache2/access.log
```

## API Methods
#### Server
http://ec2-35-157-132-97.eu-central-1.compute.amazonaws.com/

#### [GET, POST] /api/photo-prediction
http://ec2-35-157-132-97.eu-central-1.compute.amazonaws.com//api/photo-prediction

#### [GET, POST] /api/photo-prediction-mock
http://ec2-35-157-132-97.eu-central-1.compute.amazonaws.com//api/photo-prediction-mock