import tkinter.ttk
from abc import ABC
import tkinter as tk
import tkinter.messagebox
import sys


class RegistrationStepsInterface:

    def __init__(self):

        self.log_book = None
        self.bar = None
        self.text_box = None
        self.new_window = None
        self.need_evaluation = True

        self.__dict__ = {
            '0': self.load_cbct,
            '1': self.missing_tooth_localization,
            '2': self.registration,
            '3': self.save
        }

        self.step_name = ['CBCT数据导入中 ...', '缺牙牙冠识别中...', '植体配准中...', '植体保存中...']

        self.steps = 4

        self.label = None

    def load_cbct(self):
        pass

    def missing_tooth_localization(self):
        raise NotImplementedError

    def registration(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    # def eval_hooker(self):
    #
    #     self.save()
    #
    #     self.

    def run(self):

        for step in range(self.steps):

            self.label.config(text=str(step + 1) + '/' + str(self.steps) + ' ' + self.step_name[step])
            method = self.__dict__[str(step)]
            self.bar['value'] = step + 1
            self.text_box.insert(tk.END, str(step) + ': ' + self.step_name[step] + '\n')
            self.new_window.update()
            method()
            self.text_box.insert(tk.END, self.log_book + '\r\n')
            self.text_box.insert(tk.END, '\r\n')


class AbstractRegistrationMethod(RegistrationStepsInterface, ABC):

    def __init__(self, cbct_path, save_path, args):

        super(AbstractRegistrationMethod, self).__init__()

        self.new_window = tk.Tk()
        self.name = 'AbstractRegistrationMethod'
        self.new_window.title(self.name)
        self.new_window.geometry('500x300')

        self.header = tk.Label(self.new_window, text='AI辅助植牙算法正在进行：')
        self.header.pack(anchor='w', pady='15')

        self.bar = tkinter.ttk.Progressbar(self.new_window, length='300')
        self.label = tk.Label(self.new_window)
        self.text_box = tk.Text(self.new_window, height=10, width=45)

        self.cbct = cbct_path
        self.save_path = save_path
        self.status = 1
        self.args = args
        self.device = args.device
        self.pred = None

        self.log_book = ''

    def execute(self):

        self.bar.pack()

        self.label.pack(anchor='e', pady='10')

        self.text_box.pack()

        self.bar['value'], self.bar['maximum'] = self.status, 4

        try:

            self.run()

        except NotImplementedError:

            tk.messagebox.showwarning(self.new_window, message='该算法目前尚未实现')
            self.new_window.destroy()

        self.label.config(text='已保存！')

# m = AbstractRegistrationMethod('1', '', '')
# m.execute()





