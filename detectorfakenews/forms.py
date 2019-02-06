from flask_wtf import FlaskForm

from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class FormConsulta(FlaskForm):
	noticia = StringField('Cole aqui o link da notícia...', validators=[DataRequired()])
	submit = SubmitField('Analisar')