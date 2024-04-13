from flask import Flask, flash, render_template, request, redirect, session
import time
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.abspath('uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/occupancy", methods=['POST'])
def occupancy():
    time.sleep(5)
    if request.method == 'POST':
        f = request.files['file']

        if f.filename == '':
            flash('No selected file')
            return "No selected file"
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            print(filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'success'

    print(request)
    return 'fail'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS