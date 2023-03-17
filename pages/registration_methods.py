import tkinter as tk
from events.registration import MethodOne, MethodTwo, MethodThree, MethodForExample
from events.registration.threeD_dentAL_by_crown_seg import ThreeDDentALByCrownSeg
from events.registration.threeD_dentAL_by_crown_seg import ThreeDDentALByCrownSeg
from events.registration.threeD_dental_by_implant_seg import ThreeDDentALByImplantSeg


def listener(method_regist_func):

    def wrapper(self):

        self.window.destroy()

        method = method_regist_func(self)

        method.execute()

    return wrapper


class RegistrationMethodSelection:

    def __init__(self, cbct_path, save_path, args, title='植体配准算法', size='500x300'):

        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry(size)
        self.cbct_path = cbct_path
        self.save_path = save_path
        self.args = args

        self.method = None

        self.components = {
            'label': tk.Label(self.window, text='请选择配准算法'),
            'btn_method1': tk.Button(self.window, text='2.5D植体配准算法', command=self.method1_registration),
            'btn_method2': tk.Button(self.window, text='3D牙冠辅助植体配准算法', command=self.method2_registration),
            'btn_method3': tk.Button(self.window, text='3D植体配准算法', command=self.method3_registration)
        }

        self.pre_window_components = {

        }

        self.__layout()

    @listener
    def method1_registration(self):
        self.method = MethodForExample(self.cbct_path, self.save_path, self.args)
        return self.method

    @listener
    def method2_registration(self):
        self.method = ThreeDDentALByCrownSeg(self.cbct_path, self.save_path, self.args)
        return self.method

    @listener
    def method3_registration(self):

        self.method = ThreeDDentALByImplantSeg(self.cbct_path, self.save_path, self.args)
        return self.method

    def __layout(self):

        for component in self.components.values():
            component.pack(pady='20')

        self.window.mainloop()



