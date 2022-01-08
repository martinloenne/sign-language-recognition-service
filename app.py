import flask

from flask_cors import CORS
from main.web.routes import routes

app = flask.Flask(__name__)
cors = CORS(app)

app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'

app.register_blueprint(routes)

if __name__ == '__main__':
    app.run()