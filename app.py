from flask import Flask
from flask_restful import Resource, Api
from controllers import PredictResource
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'World'}

api.add_resource(HelloWorld, '/')
api.add_resource(PredictResource, '/predict')

if __name__ == "__main__":
    app.run(debug=True)