#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   PILATUS_CompareDataQt.py reads SCXRD data formats .raw, .hkl, .fcf and
#   calculates the diff / mean vs. intensity values of equivalent observation.
#   It currently reads SAINT .raw, XD2006 .fcf and general SHELX .hkl files.
#   Copyright (C) 2018, L.Krause <lkrause@chem.au.dk>, Aarhus University, DK.
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the license, or (at your option)
#   any later version.
#
#   This program is distributed in the hope that it will be useful, but WITHOUT
#   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
#   more details. <http://www.gnu.org/licenses/>
#
_REVISION = 'v2019-01-18'

from PyQt5 import uic
from PyQt5 import QtGui, QtWidgets, QtCore

import matplotlib as mpl
try:
    mpl.use('Qt5Agg')
except ImportError:
    os.environ['TCL_LIBRARY'] = '{}/tcl/tcl8.5'.format(sys.prefix)
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import numpy as np
from collections import OrderedDict
import time
import os, sys, traceback, logging

class WorkerSignals(QtCore.QObject):
    '''
    SOURCE: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(tuple)

class Worker(QtCore.QRunnable):
    '''
    SOURCE: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        logging.info(self.__class__.__name__)
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        logging.info(self.__class__.__name__)
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            r = self.fn(*self.args)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit((r, self.kwargs))
        finally:
            self.signals.finished.emit(self.kwargs)

class QLineEditDropHandler(QtCore.QObject):
    def __init__(self, parent = None):
        logging.info(self.__class__.__name__)
        QtCore.QObject.__init__(self, parent)
    
    def valid_ext(self, ext):
        #logging.info(self.__class__.__name__)
        if ext in ['.raw', '.hkl', '.fco', '.fcf', '.vsf']:
            return True
        else:
            return False
        
    def eventFilter(self, obj, event):
        #logging.info(self.__class__.__name__)
        if event.type() == QtCore.QEvent.DragEnter:
            md = event.mimeData()
            if md.hasUrls():
                for url in md.urls():
                    filePath = url.toLocalFile()
                    root, ext = os.path.splitext(filePath)
                    if self.valid_ext(ext):
                        event.accept()
        
        if event.type() == QtCore.QEvent.Drop:
            md = event.mimeData()
            if md.hasUrls():
                for url in md.urls():
                    filePath = url.toLocalFile()
                    root, ext = os.path.splitext(filePath)
                    if self.valid_ext(ext):
                        obj.clear()
                        obj.setText(filePath)
                        obj.returnPressed.emit()
                        return True
            
        return QtCore.QObject.eventFilter(self, obj, event)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        logging.info(self.__class__.__name__)
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('PCDQt_IUCr.ui', self)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.WindowCloseButtonHint)
        
        self.le_data_1.installEventFilter(QLineEditDropHandler(self))
        self.le_data_2.installEventFilter(QLineEditDropHandler(self))
        self.le_data_1.returnPressed.connect(lambda: self.prepare_read_data(self.le_data_1.text(), self.le_data_1, self.ready_data_1))
        self.le_data_1.returnPressed.connect(lambda: self.update_last_dir(self.le_data_1.text()))
        self.le_data_2.returnPressed.connect(lambda: self.prepare_read_data(self.le_data_2.text(), self.le_data_2, self.ready_data_2))
        self.le_data_2.returnPressed.connect(lambda: self.update_last_dir(self.le_data_2.text()))
        self.tb_plot.pressed.connect(self.plot_data)
        self.tb_data_1.pressed.connect(lambda: self.open_file_browser(self.le_data_1))
        self.tb_data_2.pressed.connect(lambda: self.open_file_browser(self.le_data_2))
        self.cb_sym.currentTextChanged.connect(self.set_symmetry_operations)
        self.btn_clear.pressed.connect(self.clear_all)
        
        self.HKL_1 = OrderedDict()
        self.HKL_2 = OrderedDict()
        self.threadpool = QtCore.QThreadPool()
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data_1 = None
        self.data_2 = None
        self.last_dir = None
        
        self.group_scale = QtWidgets.QButtonGroup()
        self.group_scale.addButton(self.rb_scale_1)
        self.group_scale.addButton(self.rb_scale_2)
        
        self.init_custom_styles()
        self.init_symmetry()
        
    def update_last_dir(self, aPath):
        logging.info(self.__class__.__name__)
        self.last_dir = aPath
    
    def open_file_browser(self, aWidget):
        logging.info(self.__class__.__name__)
        aPath = QFileDialog.getOpenFileName(self, 'Open File', self.last_dir, 'SCXRD Data Formats (*.raw, *.fco, *.hkl)', 'hahaha', QFileDialog.DontUseNativeDialog)[0]
        if not os.path.exists(aPath):
            return
        self.last_dir = aPath
        aWidget.setText(aPath)
        aWidget.returnPressed.emit()
    
    def init_symmetry(self):
        logging.info(self.__class__.__name__)
        self.Symmetry = {  '1':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]]]),
                          '-1':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]]]),
                         
                         '2/m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]]]),
                     
                         '222':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]]]),
                         
                         'mmm':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]]]),
                     
                         '4/m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]]]),
                     
                       '4/mmm':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0,  1]]]),
                                
                        'm-3m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  0,  0,  1],[  1,  0,  0],[  0,  1,  0]],
                                         [[  0,  0,  1],[ -1,  0,  0],[  0, -1,  0]],
                                         [[  0,  0, -1],[ -1,  0,  0],[  0,  1,  0]],
                                         [[  0,  0, -1],[  1,  0,  0],[  0, -1,  0]],
                                         [[  0,  1,  0],[  0,  0,  1],[  1,  0,  0]],
                                         [[  0, -1,  0],[  0,  0,  1],[ -1,  0,  0]],
                                         [[  0,  1,  0],[  0,  0, -1],[ -1,  0,  0]],
                                         [[  0, -1,  0],[  0,  0, -1],[  1,  0,  0]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[  1,  0,  0],[  0,  0,  1],[  0, -1,  0]],
                                         [[ -1,  0,  0],[  0,  0,  1],[  0,  1,  0]],
                                         [[ -1,  0,  0],[  0,  0, -1],[  0, -1,  0]],
                                         [[  1,  0,  0],[  0,  0, -1],[  0,  1,  0]],
                                         [[  0,  0,  1],[  0,  1,  0],[ -1,  0,  0]],
                                         [[  0,  0,  1],[  0, -1,  0],[  1,  0,  0]],
                                         [[  0,  0, -1],[  0,  1,  0],[  1,  0,  0]],
                                         [[  0,  0, -1],[  0, -1,  0],[ -1,  0,  0]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[  0,  0, -1],[ -1,  0,  0],[  0, -1,  0]],
                                         [[  0,  0, -1],[  1,  0,  0],[  0,  1,  0]],
                                         [[  0,  0,  1],[  1,  0,  0],[  0, -1,  0]],
                                         [[  0,  0,  1],[ -1,  0,  0],[  0,  1,  0]],
                                         [[  0, -1,  0],[  0,  0, -1],[ -1,  0,  0]],
                                         [[  0,  1,  0],[  0,  0, -1],[  1,  0,  0]],
                                         [[  0, -1,  0],[  0,  0,  1],[  1,  0,  0]],
                                         [[  0,  1,  0],[  0,  0,  1],[ -1,  0,  0]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0,  0, -1],[  0,  1,  0]],
                                         [[  1,  0,  0],[  0,  0, -1],[  0, -1,  0]],
                                         [[  1,  0,  0],[  0,  0,  1],[  0,  1,  0]],
                                         [[ -1,  0,  0],[  0,  0,  1],[  0, -1,  0]],
                                         [[  0,  0, -1],[  0, -1,  0],[  1,  0,  0]],
                                         [[  0,  0, -1],[  0,  1,  0],[ -1,  0,  0]],
                                         [[  0,  0,  1],[  0, -1,  0],[ -1,  0,  0]],
                                         [[  0,  0,  1],[  0,  1,  0],[  1,  0,  0]]])}
        
        [self.cb_sym.addItem(i) for i in sorted(self.Symmetry.keys())]
        self.cb_sym.setCurrentText('1')
    
    def set_symmetry_operations(self):
        logging.info(self.__class__.__name__)
        self.SymOp = self.Symmetry[self.cb_sym.currentText()]
        self.HKL_1 = OrderedDict()
        self.HKL_2 = OrderedDict()
        self.la_data_sym.setText('-')
        if self.data_1 is not None and self.data_2 is not None:
            self.thread_run(self.dict_symmetry_equivalents, self.data_1, self.HKL_1, 'Io_1', 'Is_1', flag = 'ready_data_1')
            self.thread_run(self.dict_symmetry_equivalents, self.data_2, self.HKL_2, 'Io_2', 'Is_2', flag = 'ready_data_2')
        
    def init_custom_styles(self):
        logging.info(self.__class__.__name__)
        self.tb_style = ('QToolButton          {background-color: rgb(240, 250, 240); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75); border-radius: 5px}'
                         'QToolButton:hover    {background-color: rgb(250, 255, 250); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:pressed  {background-color: rgb(255, 255, 255); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:checked  {background-color: rgb(220, 220, 220); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:disabled {background-color: rgb(220, 200, 200); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}')
        
        self.tb_plot.setStyleSheet(self.tb_style)

    def prepare_read_data(self, aPath, aWidget, aFlag):
        logging.info(self.__class__.__name__)
        if not os.path.exists(aPath):
            print('invalid path!')
            return
        aFlag = False
        aWidget.setEnabled(False)
        self.thread_run(self.read_data, aPath, parent_widget = aWidget)
        
    def read_data(self, fname, use_columns = None, used_only = True):
        logging.info(self.__class__.__name__)
        '''
        
        '''
        name, ext = os.path.splitext(fname)
        if ext == '.raw':
            if not use_columns:
                use_columns = (0,1,2,3,4)
            data = np.genfromtxt(fname, usecols=use_columns, delimiter=[4,4,4,8,8,4,8,8,8,8,8,8,3,7,7,8,7,7,8,6,5,7,7,7,2,5,9,7,7,4,6,11,3,6,8,8,8,8,4])
        elif ext == '.fco':
            # delimiter=[6,5,5,11,11,11,11,4])
            # skip_header=26
            if not use_columns:
                use_columns = (0,1,2,4,5,6,7)
            data = np.genfromtxt(fname, skip_header=26, usecols=use_columns)
            if used_only:
                data = data[data[::,6] == 0]
            data = data[:,[0,1,2,3,4,5]]
        elif ext == '.fcf':
            if not use_columns:
                use_columns = (0,1,2,4,5)
            data = np.genfromtxt(fname, skip_header=78, usecols=use_columns)
        elif ext == '.sortav':
            if not use_columns:
                use_columns = (0,1,2,3,6)
            data = np.genfromtxt(fname, usecols=use_columns, comments='c')
        elif ext == '.vsf':
            if not use_columns:
                use_columns = (0,1,2,6)
            data = np.genfromtxt(fname, skip_header=1, usecols=use_columns)
            data = np.hstack([data, data[::,3][:,np.newaxis]])
            data[::,3] *= data[::,3]
        elif ext == '.hkl':
            with open(fname) as ofile:
                temp = ofile.readline()
            if len(temp.split()) == 4 and 'NDAT' in temp.upper():
                # XD2006
                # HEADER:XDNAME F^2 NDAT 7
                # delimiter=[4,4,4,2,8,8,8])
                if not use_columns:
                    use_columns = (0,1,2,4,5)
                data = np.genfromtxt(fname, skip_header=1, usecols=use_columns)
            else:
                # SHELX
                # delimiter=[4,4,4,8,8,4]
                # skip_footer=17
                if not use_columns:
                    use_columns = (0,1,2,3,4)
                data = np.genfromtxt(fname, skip_footer=17, usecols=use_columns, delimiter=[4,4,4,8,8,4])
        else:
            data = None
        return data
    
    def dict_symmetry_equivalents(self, data, HKL, key_Io, key_Is):
        logging.info(self.__class__.__name__)
        '''
         TO_CHECK: can loops be merged?
        '''
        use_stl = False
        for r in data:
            h, k, l, Io, Is = r[:5]
            if len(r) == 6:
                use_stl = True
                stl = r[5]
            hkl = tuple(np.unique(np.array([h,k,l]).dot(self.SymOp), axis=0)[0])
            if hkl in HKL:
                if key_Io in HKL[hkl]:
                    HKL[hkl][key_Io].append(Io)
                    HKL[hkl][key_Is].append(Is)
                else:
                    HKL[hkl][key_Io] = [Io]
                    HKL[hkl][key_Is] = [Is]
            else:
                if use_stl:
                    HKL[hkl] = {key_Io:[Io], key_Is:[Is], 'stl':stl}
                else:
                    HKL[hkl] = {key_Io:[Io], key_Is:[Is]}
    
    def calculate_statistics(self):
        logging.info(self.__class__.__name__)
        multi = []
        meaIo = []
        medIo = []
        rIsig = []
        rIstd = []
        hkl   = []
        stl   = []
        for (h,k,l) in self.HKL_1:
            if (h,k,l) in self.HKL_2:
                Io_mean_1 = np.mean(self.HKL_1[(h,k,l)]['Io_1'])
                Io_mean_2 = np.mean(self.HKL_2[(h,k,l)]['Io_2'])
                Io_medi_1 = np.median(self.HKL_1[(h,k,l)]['Io_1'])
                Io_medi_2 = np.median(self.HKL_2[(h,k,l)]['Io_2'])
                Io_std_1  = np.std(self.HKL_1[(h,k,l)]['Io_1'])
                Io_std_2  = np.std(self.HKL_2[(h,k,l)]['Io_2'])
                Is_mean_1 = np.mean(self.HKL_1[(h,k,l)]['Is_1'])
                Is_mean_2 = np.mean(self.HKL_2[(h,k,l)]['Is_2'])
                if 'stl' in self.HKL_1[(h,k,l)]:
                    stl.append(self.HKL_1[(h,k,l)]['stl'])
                multi.append((len(self.HKL_1[(h,k,l)]['Io_1']), len(self.HKL_2[(h,k,l)]['Io_2'])))
                meaIo.append((Io_mean_1, Io_mean_2))
                medIo.append((Io_medi_1, Io_medi_2))
                rIsig.append((Io_mean_1 / Is_mean_1, Io_mean_2 / Is_mean_2))
                rIstd.append((Io_mean_1 / Io_std_1, Io_mean_2 / Io_std_2))
                hkl.append((h,k,l))
            else:
                print('> unmatched: ({:3}{:3}{:3}) {}'.format(int(h), int(k), int(l), self.HKL_1[(h,k,l)]))

        self.multi = np.asarray(multi)
        self.meaIo = np.asarray(meaIo)
        self.medIo = np.asarray(medIo)
        self.rIsig = np.asarray(rIsig)
        self.rIstd = np.asarray(rIstd)
        self.hkl   = np.asarray(hkl)
        self.stl   = np.asarray(stl)
    
    def plot_data(self):
        logging.info(self.__class__.__name__)
        _SIGCUT  = round(self.db_sigcut.value(), 1)
        _SCALE   = self.rb_scale_1.isChecked()
        _SAVE    = self.cb_save.isChecked()
        _FILE_1  = self.le_data_1.text()
        _FILE_2  = self.le_data_2.text()
        _LABEL_1 = self.le_label_1.text()
        _LABEL_2 = self.le_label_2.text()
        _PREFIX  = self.le_prefix.text()
        _TITLE   = self.cb_title.isChecked()
        
        mpl.rcParams['figure.figsize']   = [13.66, 7.68]
        #mpl.rcParams['figure.dpi']      = 600
        mpl.rcParams['savefig.dpi']      = 300
        mpl.rcParams['font.size']        = 11
        mpl.rcParams['legend.fontsize']  = 11
        mpl.rcParams['figure.titlesize'] = 11
        mpl.rcParams['figure.titlesize'] = 11
        mpl.rcParams['axes.titlesize']   = 11
        mpl.rcParams['axes.labelsize']   = 11
        mpl.rcParams['lines.linewidth']  = 1
        mpl.rcParams['lines.markersize'] = 4
        mpl.rcParams['xtick.labelsize']  = 8
        mpl.rcParams['ytick.labelsize']  = 8
        scatter_marker_size = 12
        
        fig = plt.figure()
        grid = plt.GridSpec(10, 16, wspace=0.0, hspace=0.0)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.12)
        
        sigcut = _SIGCUT
        scale = round(self.ds_scale.value(), 3)
        if _SCALE:
            scale = round(np.nansum(np.prod(self.meaIo, axis=1))/np.nansum(np.square(self.meaIo[:,0])), 3)
        
        if _TITLE:
            fig.suptitle('Scalefactor: {:6.3f}, cutoff: {}, symmetry: {}\n1: {}\n2: {}'.format(scale, sigcut, self.cb_sym.currentText(), _FILE_1, _FILE_2))
            fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.12)
            
        f1 = self.meaIo[:,0]*scale
        f2 = self.meaIo[:,1]
        rIsig = self.rIsig
        
        f1cut = f1[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)]
        f2cut = f2[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)]
        
        p00 = fig.add_subplot(grid[ :2,  :7])
        p01 = fig.add_subplot(grid[ :2, 9: ])
        p1x = fig.add_subplot(grid[4:9, 1: ])
        h1y = fig.add_subplot(grid[4:9, 0  ], sharey=p1x)
        h1x = fig.add_subplot(grid[9  , 1: ], sharex=p1x)
        
        p00.scatter(f1cut, f2cut, s=scatter_marker_size, color='#37A0CB')
        p00.plot([0, np.nanmax(f1cut)],[0, np.nanmax(f1cut)], 'k-', lw=1.0)
        p00.set_xlabel(r'$I_{{{}}}$'.format(_LABEL_1))
        p00.set_ylabel(r'$I_{{{}}}$'.format(_LABEL_2))
        p00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        p00.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        x = np.log10(f1cut)
        y = np.log10(f2cut)
        p01.scatter(x, y, s=scatter_marker_size, color='#37A0CB')
        p01.plot([np.nanmin(x), np.nanmax(x)],[np.nanmin(x), np.nanmax(x)], 'k-', lw=1.0)
        p01.set_xlabel(r'$\log\left(I_{{{}}}\right)$'.format(_LABEL_1))
        p01.set_ylabel(r'$\log\left(I_{{{}}}\right)$'.format(_LABEL_2))
        
        facut = (f1cut + f2cut) / 2.
        x = np.log10(facut)
        
        y = (f1cut - f2cut)/(facut)
        
        p1x_sc = p1x.scatter(x, y, s=scatter_marker_size, alpha=1.0, picker=True, color='#37A0CB')
        p1x.plot([np.min(x), np.max(x)], [0,0], 'k-', lw=1.0)
        p1x.xaxis.set_visible(False)
        p1x.yaxis.set_visible(False)
        
        h1y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
        h1y.xaxis.set_visible(False)
        h1y.invert_xaxis()
        h1y.spines['top'].set_visible(False)
        h1y.spines['bottom'].set_visible(False)
        h1y.set_ylabel(r'$\left(I_{{{0:}}}\ -\ I_{{{1:}}}\right)\ /\ \left<I_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_LABEL_1, _LABEL_2))
        
        h1x.hist(x[~np.isnan(x)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
        h1x.yaxis.set_visible(False)
        h1x.spines['left'].set_visible(False)
        h1x.spines['right'].set_visible(False)
        h1x.invert_yaxis()
        h1x.set_xlabel(r'$\log\left(\left<I_{{{{{0:}}},{{{1:}}}}}\right>\right)$'.format(_LABEL_1, _LABEL_2))

        if _SAVE:
            pname = r'{}_{}_vs_{}_c{}_s{}'.format(_PREFIX, _LABEL_1.replace('\\', ''), _LABEL_2.replace('\\', ''), sigcut, scale)
            #plt.savefig(pname + '.pdf', transparent=True)
            plt.savefig(pname + '.png', dpi=600, transparent=True)
        
        from collections import defaultdict
        self.annotations = defaultdict(list)
        self.background = fig.canvas.copy_from_bbox(p1x.bbox)
        def on_pick(event):
            x = event.mouseevent.x
            y = event.mouseevent.y
            ind = event.ind[0]
            h,k,l = map(int, self.hkl[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)][ind])
            ann_name = '{:3}{:3}{:3}'.format(h,k,l)
            
            if (event.mouseevent.button == 3 and len(self.annotations) > 0) or ann_name in self.annotations:
                self.annotations[ann_name].remove()
                self.annotations.pop(ann_name)
                fig.canvas.draw_idle()
                return
                
            self.annotations[ann_name] = plt.annotate(ann_name, xy=(x,y), size=8, xycoords='figure pixels')
            print(ann_name)
            #fig.canvas.draw_idle()
            p1x.draw_artist(self.annotations[ann_name])
        
        def update_annot(ind, pos, but):
            h,k,l = map(int, self.hkl[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)][ind])
            ann_name = '{:3}{:3}{:3}'.format(h,k,l)
            
            if ann_name in self.annotations:
                if but == 3:
                    self.annotations[ann_name].remove()
                    self.annotations.pop(ann_name)
                return
            elif but == 1:
                self.annotations[ann_name] = plt.annotate(ann_name, xy=pos, size=8, xycoords='figure pixels')
                print(ann_name)
            
        def hover(event):
            if event.inaxes == p1x:
                cont, ind = p1x_sc.contains(event)
                if cont:
                    update_annot(ind["ind"][0], (event.x, event.y), event.button)
                    fig.canvas.draw_idle()
                    
        fig.canvas.mpl_connect('motion_notify_event', hover)
        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()
            
    def on_thread_result(self, r):
        logging.info(self.__class__.__name__)
        data, kwargs = r
        if data is not None:
            if 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_1:
                self.data_1 = data
                self.la_data_1.setText('Reflections: {}'.format(str(len(data))))
                self.thread_run(self.dict_symmetry_equivalents, self.data_1, self.HKL_1, 'Io_1', 'Is_1', flag = 'ready_data_1')
            elif 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_2:
                self.data_2 = data
                self.la_data_2.setText('Reflections: {}'.format(str(len(data))))
                self.thread_run(self.dict_symmetry_equivalents, self.data_2, self.HKL_2, 'Io_2', 'Is_2', flag = 'ready_data_2')

    def clear_all(self):
        self.le_data_1.setText('')
        self.le_data_2.setText('')
        self.la_data_1.setText('-')
        self.la_data_2.setText('-')
        self.la_data_sym.setText('-')
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data_1 = None
        self.data_2 = None
        self.last_dir = None
        self.HKL_1.clear()
        self.HKL_2.clear()
        self.hkl = []
        
    def on_thread_finished(self, kwargs):
        logging.info(self.__class__.__name__)
        if self.threadpool.activeThreadCount() == 0:
            self.statusBar.showMessage('ready.')
        if 'flag' in kwargs:
            if kwargs['flag'] == 'ready_data_1':
                self.le_data_1.setEnabled(True)
                self.ready_data_1 = True
                if self.ready_data_1 and self.ready_data_2:
                    self.thread_run(self.calculate_statistics, flag = 'ready_data_all')
            elif kwargs['flag'] == 'ready_data_2':
                self.le_data_2.setEnabled(True)
                self.ready_data_2 = True
                if self.ready_data_1 and self.ready_data_2:
                    self.thread_run(self.calculate_statistics, flag = 'ready_data_all')
            elif kwargs['flag'] == 'ready_data_all' and self.threadpool.activeThreadCount() == 0:
                #####################################
                ## calculations are finished here! ##
                #####################################
                self.tb_plot.setEnabled(True)
                self.cb_sym.setEnabled(True)
                self.la_data_sym.setText(str(len(self.hkl)))
    
    def thread_run(self, fn, *args, **kwargs):
        logging.info(self.__class__.__name__)
        self.tb_plot.setEnabled(False)
        w = Worker(fn, *args, **kwargs)
        w.signals.result.connect(self.on_thread_result)
        w.signals.finished.connect(self.on_thread_finished)
        self.threadpool.start(w)
        ## ALWAYS DIASBLE THE SYM SWITCH ##
        self.cb_sym.setEnabled(False)
        ###################################
        self.statusBar.showMessage('I\'m thinking ...')
        if 'parent_widget' in kwargs:
            kwargs['parent_widget'].setEnabled(False)

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.setWindowTitle('Compare SCXRD Data, {}'.format(_REVISION))
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    # Remove existing handlers, Python creates a
    # default handler that goes to the console
    # and will ignore further basicConfig calls
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    # create logger
    logging.basicConfig(level=logging.INFO, format='%(message)20s > %(funcName)s')
    main()