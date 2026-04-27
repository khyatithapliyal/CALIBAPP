# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-16 13:09:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-11-09 18:41:26

import tkinter as tk
from tkinter import filedialog, messagebox


def open_file_dialog(filetype, dirname='', title='Open a file', multiple=False):
    '''
    Open a FileOpenDialogBox to select file(s). The default directory and file type are given.

    :param dirname: default directory
    :param filetype: default file type
    :param title: dialog window title
    :param multiple: whether or not to allow for multiple files
    :return: full path to the chosen file (or None)
    '''
    root = tk.Tk()
    root.withdraw()
    kwargs = dict(
        master=root,
        title=title,
        filetypes=[(f'{filetype} files', f'.{filetype}')],
        initialdir=dirname
    )
    if multiple:
        fpath = filedialog.askopenfilenames(**kwargs)
    else:
        fpath = filedialog.askopenfilename(**kwargs)
    root.destroy()
    if len(fpath) == 0:
        return None
    return fpath


def open_folder_dialog(title='Select folder', **kwargs):
    '''
    Open a dialog box to select directory.

    :param title: dialog window title
    :return: full path to the chosen directory (or None)
    '''
    root = tk.Tk()
    root.withdraw()
    dirpath = filedialog.askdirectory(**kwargs)
    root.destroy()
    if len(dirpath) == 0:
        return None
    return dirpath


def multi_select_dialog(candidates, title='Select options'):
    '''
    Open a dialog window to select 1 or multiple options from a list of candidates
    
    :param candidates: list of options
    :return: list of selected options
    '''
    root = tk.Tk()
    root.title(title)
    listbox = tk.Listbox(root, selectmode='multiple', height=len(candidates))
    listbox.pack()
    for candidate in candidates:
        listbox.insert(tk.END, candidate)
    my_sel = None
    def get_selected():
        global my_sel
        my_sel = [listbox.get(i) for i in listbox.curselection()]
        root.destroy()
    sel_button = tk.Button(root, text='OK', command=get_selected)
    sel_button.pack()
    root.mainloop()
    print(my_sel)
    return my_sel


def askyesno_dialog(s):
    ''' 
    Show a dialog windonw asking the user a yes or no question

    :param s: yes/no question
    :return: user answer (True for yes, False for no)
    '''
    root = tk.Tk()
    root.withdraw()
    out = messagebox.askyesno('Question', s)
    root.destroy()
    return out

