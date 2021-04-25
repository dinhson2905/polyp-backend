from flask_restful import Resource

class BaseResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__()
