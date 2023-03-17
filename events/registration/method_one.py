from events.registration.methods import AbstractRegistrationMethod


class MethodOne(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path, args):

        super(MethodOne, self).__init__(cbct_path, save_path, args)

