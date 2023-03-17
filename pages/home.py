import tkinter as tk
from pages.registration_methods import RegistrationMethodSelection
from tkinter.filedialog import askopenfilename, askdirectory, askopenfilenames
import tkinter.messagebox
import argparse
import sys
sys.path.append('events')

class MainPage:

    def __init__(self, args, title='DentAI 计算机辅助植牙界面', size='500x300'):

        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry(size)

        self.components = {'btn_cbct': tk.Button(self.window, text='选择CBCT口腔文件', command=self.__select_cbct_file),
                           'btn_save_path': tk.Button(self.window, text='选择保存路径', command=self.__select_save_path),
                           'cbct_path': tk.Label(self.window), 'save_path': tk.Label(self.window),
                           'btn_continue': tk.Button(self.window, text='继续', command=self.__march),
                           }
        self.args = args

        self.dicom_path = ''
        self.__layout()

    def __select_cbct_file(self):

        path = askopenfilename()
        self.components['cbct_path'].config(text=path)

    def __select_save_path(self):

        path = askdirectory()
        self.components['save_path'].config(text=path)

    def __march(self):

        cbct_path = self.components['cbct_path']['text']
        save_path = self.components['save_path']['text']

        alert = ''
        if cbct_path == '':
            alert = '没有指定口腔CBCT文件'

        if alert != '':
            self.__pop_window(alert)
            return

        # create the second window for methods selection
        method_selection = RegistrationMethodSelection(cbct_path, save_path, self.args)


    def __pop_window(self, alert):

        tk.messagebox.showwarning(self.window, message=alert)

    def __layout(self):

        self.components['btn_cbct'].pack(pady='10')

        self.components['cbct_path'].pack()

        self.components['btn_save_path'].pack(pady='10')

        self.components['save_path'].pack()

        self.components['btn_continue'].pack(pady='5')

        self.window.mainloop()

