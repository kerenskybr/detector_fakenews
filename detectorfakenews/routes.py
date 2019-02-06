from detectorfakenews import app

from flask import render_template, url_for

from detectorfakenews.forms import FormConsulta

@app.route('/', methods=['GET', 'POST'])
def index():




	return render_template('index.html')