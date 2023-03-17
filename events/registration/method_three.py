from abc import ABC

from events.registration.methods import AbstractRegistrationMethod


class MethodThree(AbstractRegistrationMethod, ABC):

    def __init__(self, cbct_path, implant_path, save_path):

        super(MethodThree, self).__init__(cbct_path, implant_path, save_path)
