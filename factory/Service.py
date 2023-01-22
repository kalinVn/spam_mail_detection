import config
from service.SpamMailDetection import SpamMailDetection


class Service:

    def __init__(self):
        self.service_type = config.SERVICE_TYPE

    def get_service(self):
        if self.service_type == "ML":
            return SpamMailDetection()

