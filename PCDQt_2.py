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
import pandas as pd
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
    def __init__(self, parent=None):
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
        uic.loadUi('PCDQt.ui', self)
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
        
        self.threadpool = QtCore.QThreadPool()
        self.ready_data = False
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data_1 = None
        self.data_2 = None
        self.data = None
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
        aPath = QFileDialog.getOpenFileName(self, 'Open File', self.last_dir, 'SCXRD Data Formats (*.raw, *.fco, *.hkl)', '-', QFileDialog.DontUseNativeDialog)[0]
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
        self.la_data_sym.setText('-')
        if self.ready_data_1 and self.ready_data_2:
            self.merge_data()
        
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
        self.thread_run(self.read_data, aPath, parent_widget=aWidget)
    
    def read_data(self, fname, use_columns=None, used_only=True):
        logging.info(self.__class__.__name__)
        '''
        
        '''
        name, ext = os.path.splitext(fname)
        ints = ['h','k','l']
        floats = ['Fo','Fs']
        self.use_stl = False
        
        if ext == '.raw':
            if not use_columns:
                use_columns = (0,1,2,3,4)
            raw_data = np.genfromtxt(fname, dtype=float, usecols=use_columns, delimiter=[4,4,4,8,8,4,8,8,8,8,8,8,3,7,7,8,7,7,8,6,5,7,7,7,2,5,9,7,7,4,6,11,3,6,8,8,8,8,4])
        elif ext == '.fco':
            # delimiter=[6,5,5,11,11,11,11,4])
            # skip_header=26
            if not use_columns:
                use_columns = (0,1,2,4,5,6,7)
            raw_data = np.genfromtxt(fname, dtype=float, skip_header=26, usecols=use_columns)
            if used_only:
                raw_data = raw_data[raw_data[::,6] == 0]
            floats = ['Fo','Fs','stl']
            self.use_stl = True
            raw_data = raw_data[:,[0,1,2,3,4,5]]
        elif ext == '.sortav':
            if not use_columns:
                use_columns = (0,1,2,3,6)
            raw_data = np.genfromtxt(fname, dtype=float, usecols=use_columns, comments='c')
        elif ext == '.hkl':
            with open(fname) as ofile:
                temp = ofile.readline()
            if len(temp.split()) == 4 and 'NDAT' in temp:
                # XD2006
                # HEADER:XDNAME F^2 NDAT 7
                # delimiter=[4,4,4,2,8,8,8])
                if not use_columns:
                    use_columns = (0,1,2,4,5)
                raw_data = np.genfromtxt(fname, dtype=float, skip_header=1, usecols=use_columns)
            else:
                # SHELX
                # delimiter=[4,4,4,8,8,4]
                # skip_footer=17
                if not use_columns:
                    use_columns = (0,1,2,3,4)
                raw_data = np.genfromtxt(fname, dtype=float, skip_footer=17, usecols=use_columns, delimiter=[4,4,4,8,8,4])
        else:
            data = None
        data = pd.DataFrame(raw_data, columns=ints+floats)
        data = data.astype(dict(zip(ints,[int]*len(ints))))
        return data
    
    def merge_data(self):
        logging.info(self.__class__.__name__)

        # find common base for hkl
        def reducehkltofam(hkl, SymOps):
            return tuple(np.unique((hkl.to_numpy()).dot(SymOps), axis=0)[-1])
        # Reduce hkls according to symmetry
        self.data_1['base'] = self.data_1[['h','k','l']].apply(reducehkltofam, args=(self.SymOp,), axis=1)
        self.data_2['base'] = self.data_2[['h','k','l']].apply(reducehkltofam, args=(self.SymOp,), axis=1)
        
        self.data = self.data_1.merge(self.data_2, how='inner', on=['base','h','k','l'], suffixes=('_1','_2'), indicator=True)
        self.la_data_sym.setText(str(len(set(self.data['base']))))
        
        self.ready_data = True
    
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
        mpl.rcParams['font.size']        = 12
        mpl.rcParams['legend.fontsize']  = 12
        mpl.rcParams['figure.titlesize'] = 12
        mpl.rcParams['figure.titlesize'] = 12
        mpl.rcParams['axes.titlesize']   = 12
        mpl.rcParams['axes.labelsize']   = 12
        mpl.rcParams['lines.linewidth']  = 1
        mpl.rcParams['lines.markersize'] = 8
        mpl.rcParams['xtick.labelsize']  = 8
        mpl.rcParams['ytick.labelsize']  = 8
        scatter_marker_size = 12
        
        fig = plt.figure()
        if self.use_stl:
            grid = plt.GridSpec(12, 13, wspace=0.0, hspace=0.0)
        else:
            grid = plt.GridSpec(7, 13, wspace=0.0, hspace=0.0)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, wspace=0.0, hspace=0.0)
        
        grouped = self.data.groupby(['base'])
        f1cut = grouped['Fo_1'].transform(np.mean)
        f2cut = grouped['Fo_2'].transform(np.mean)
        
        sigcut = _SIGCUT
        scale = self.ds_scale.value()
        if _SCALE:
            scale = np.nansum(f1cut*f2cut)/np.nansum(np.square(f1cut))
        
        if _TITLE:
            fig.suptitle('Scalefactor: {:6.3f}, cutoff: {}, symmetry: {}\n1: {}\n2: {}'.format(scale, sigcut, self.cb_sym.currentText(), _FILE_1, _FILE_2))
            fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.12)
        
        f1cut *= scale
        
        if self.use_stl:
            p00 = fig.add_subplot(grid[ :2 ,  :6])
            p01 = fig.add_subplot(grid[ :2 , 7: ])
            p1x = fig.add_subplot(grid[3:6 , 1: ])
            h1y = fig.add_subplot(grid[3:6 , 0  ], sharey=p1x)
            h1x = fig.add_subplot(grid[6   , 1: ], sharex=p1x)
            p2x = fig.add_subplot(grid[8:11, 1: ])
            h2y = fig.add_subplot(grid[8:11, 0  ], sharey=p2x)
            h2x = fig.add_subplot(grid[11  , 1: ], sharex=p2x)
            mpl.rcParams['figure.figsize'] = [13.66, 10.24]
        else:
            p00 = fig.add_subplot(grid[ :2,  :6])
            p01 = fig.add_subplot(grid[ :2, 7: ])
            p1x = fig.add_subplot(grid[3:6, 1: ])
            h1y = fig.add_subplot(grid[3:6, 0  ], sharey=p1x)
            h1x = fig.add_subplot(grid[6  , 1: ], sharex=p1x)
        
        p00.scatter(f1cut, f2cut, s=4, color='#37A0CB')
        p00.plot([0, np.nanmax(f1cut)],[0, np.nanmax(f1cut)], 'k-', lw=1.0)
        p00.set_xlabel(r'$I_{{{}}}$'.format(_LABEL_1))
        p00.set_ylabel(r'$I_{{{}}}$'.format(_LABEL_2))
        p00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        p00.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        x = np.log10(f1cut)
        y = np.log10(f2cut)
        p01.scatter(x, y, s=4, color='#37A0CB')
        p01.plot([np.nanmin(x), np.nanmax(x)],[np.nanmin(x), np.nanmax(x)], 'k-', lw=1.0)
        p01.set_xlabel(r'$\log\left(I_{{{}}}\right)$'.format(_LABEL_1))
        p01.set_ylabel(r'$\log\left(I_{{{}}}\right)$'.format(_LABEL_2))
        
        facut = (f1cut + f2cut) / 2.
        x = np.log10(facut)
        y = (f1cut - f2cut)/(facut)
        
        p1x_sc = p1x.scatter(x, y, s=20, alpha=0.5, picker=True, color='#37A0CB')
        p1x.plot([np.min(x), np.max(x)], [0,0], 'k-', lw=1.0)
        p1x.spines['left'].set_visible(False)
        p1x.spines['bottom'].set_visible(False)
        p1x.xaxis.set_visible(False)
        p1x.yaxis.set_visible(False)
        
        if self.use_stl:
            self.data['stl'] = self.data[['stl_1', 'stl_2']].mean(axis=1)
            self.data.drop(columns=['stl_1','stl_2'], inplace=True)
            stl = self.data['stl'][(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)]
            p2x.scatter(stl, y, s=20, alpha=0.5, picker=True, color='#37A0CB')
            p2x.plot([np.min(stl), np.max(stl)], [0,0], 'k-', lw=1.0)
            p2x.set_ylabel(r'$(I_{1}\ -\ I_{2})\ /\ \left<I_{1,2}\right>$')
            p2x.spines['left'].set_visible(False)
            p2x.spines['bottom'].set_visible(False)
            p2x.yaxis.set_visible(False)
            p2x.xaxis.set_visible(False)
            
            h2y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
            h2y.xaxis.set_visible(False)
            h2y.invert_xaxis()
            h2y.spines['top'].set_visible(False)
            h2y.spines['bottom'].set_visible(False)
            h2y.set_ylabel(r'$\left(I_{{{0:}}}\ -\ I_{{{1:}}}\right)\ /\ \left<I_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_LABEL_1, _LABEL_2))
        
            h2x.hist(stl[~np.isnan(stl)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
            h2x.yaxis.set_visible(False)
            h2x.spines['left'].set_visible(False)
            h2x.spines['right'].set_visible(False)
            h2x.invert_yaxis()
            h2x.set_xlabel(r'$sin(\theta)/\lambda$')
            
        h1y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
        #h1y.set_ylim([-2.0, 2.0])
        h1y.xaxis.set_visible(False)
        h1y.invert_xaxis()
        h1y.spines['top'].set_visible(False)
        h1y.spines['bottom'].set_visible(False)
        #h1y.spines['right'].set_visible(False)
        h1y.set_ylabel(r'$\left(I_{{{0:}}}\ -\ I_{{{1:}}}\right)\ /\ \left<I_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_LABEL_1, _LABEL_2))
        
        h1x.hist(x[~np.isnan(x)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
        h1x.yaxis.set_visible(False)
        #h1x.spines['top'].set_visible(False)
        h1x.spines['left'].set_visible(False)
        h1x.spines['right'].set_visible(False)
        h1x.invert_yaxis()
        h1x.set_xlabel(r'$\log\left(\left<I_{{{{{0:}}},{{{1:}}}}}\right>\right)$'.format(_LABEL_1, _LABEL_2))
        
        if _SAVE:
            pname = r'{}_{}_vs_{}_c{}_s{}'.format(_PREFIX, _LABEL_1.replace('\\', ''), _LABEL_2.replace('\\', ''), sigcut, scale)
            plt.savefig(pname + '.pdf', transparent=True)
            #plt.savefig(pname + '.png', dpi=600, transparent=True)
        
        plt.show()

    def clear_all(self):
        self.le_data_1.setText('')
        self.le_data_2.setText('')
        self.la_data_1.setText('-')
        self.la_data_2.setText('-')
        self.la_data_sym.setText('-')
        self.ready_data = False
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data = None
        self.data_1 = None
        self.data_2 = None
        self.last_dir = None
        
    def on_thread_result(self, r):
        logging.info(self.__class__.__name__)
        data, kwargs = r
        if data is not None:
            if 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_1:
                self.data_1 = data
                self.ready_data_1 = True
                self.la_data_1.setText('Reflections: {}'.format(str(len(data))))
            elif 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_2:
                self.data_2 = data
                self.ready_data_2 = True
                self.la_data_2.setText('Reflections: {}'.format(str(len(data))))
    
    def on_thread_finished(self, kwargs):
        logging.info(self.__class__.__name__)
        if self.threadpool.activeThreadCount() == 0:
            self.statusBar.showMessage('ready.')
            self.setEnabled(True)
        
        if self.ready_data_1 and self.ready_data_2 and not self.ready_data:
            self.thread_run(self.merge_data, flag='ready_data_all')
        
        if 'flag' in kwargs and kwargs['flag'] == 'ready_data_all' and self.threadpool.activeThreadCount() == 0:
            self.tb_plot.setEnabled(True)
            self.cb_sym.setEnabled(True)
            self.la_data_sym.setText(str(len(set(self.data['base']))))
    
    def thread_run(self, fn, *args, **kwargs):
        logging.info(self.__class__.__name__)
        self.tb_plot.setEnabled(False)
        w = Worker(fn, *args, **kwargs)
        w.signals.result.connect(self.on_thread_result)
        w.signals.finished.connect(self.on_thread_finished)
        self.threadpool.start(w)
        ## ALWAYS DIASBLE THE SYM SWITCH ##
        self.cb_sym.setEnabled(False)
        self.setEnabled(False)
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

class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

if __name__ == '__main__':
    # create logger
    logging.basicConfig(level=logging.INFO, format='%(message)20s > %(funcName)s')
    main()