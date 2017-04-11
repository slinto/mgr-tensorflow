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
## API Methods
#### Server
https://slinto-mgr-tensorflow.herokuapp.com/

#### [GET, POST] /api/photo-prediction
https://slinto-mgr-tensorflow.herokuapp.com/api/photo-prediction

#### [GET, POST] /api/photo-prediction-mock
https://slinto-mgr-tensorflow.herokuapp.com/api/photo-prediction-mock