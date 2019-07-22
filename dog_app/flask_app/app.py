from flask import Flask

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)

# set secret key
app.secret_key = 'secret key'
# specify file upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# specify maximum size of file to be uploaded
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
