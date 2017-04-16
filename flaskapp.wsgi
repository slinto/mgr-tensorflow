import sys
import os
sys.path.insert(0, '/var/www/html/flaskapp')

os.environ['IS_SERVER'] = 1

from flaskapp import app as application