from flask_restful import reqparse
from .base_controller import BaseResource
from services.predict_service import PredictService

class PredictResource(BaseResource):
    def get(self):
        return {'message': 'Get predict'}

    def post(self):
        # request: model_name, name, base64_encode
        parser = reqparse.RequestParser()
        parser.add_argument('model_name', type=str, required=True)
        parser.add_argument('name', type=str, required=True)
        parser.add_argument('base64_encode', type=str, required=True)
        args = parser.parse_args()
        
        predict_service = PredictService()
        model_name = args['model_name']
        name = args['name']
        base64_encode = args['base64_encode']
        res = predict_service.predict(model_name, name, base64_encode)
        return res
