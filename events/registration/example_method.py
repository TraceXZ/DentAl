from events.registration.methods import AbstractRegistrationMethod
import time


class MethodForExample(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path, args):

        super(MethodForExample, self).__init__(cbct_path, save_path, args)

    def missing_tooth_localization(self):
        time.sleep(2)

    def load_cbct(self):
        time.sleep(2)

    def registration(self):
        time.sleep(2)

    def save(self):
        time.sleep(2)
