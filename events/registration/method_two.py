from events.registration.methods import AbstractRegistrationMethod


class MethodTwo(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path):

        super(MethodTwo, self).__init__(cbct_path, save_path, args)

