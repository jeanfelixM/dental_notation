# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# Edited by: ME 


import pickle
import uuid
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QTextEdit, QPushButton
from pyvistaqt import BackgroundPlotter
from collections import defaultdict
import pyvista as pv
import re, os
import subprocess
import sys
from os import path
from alig.convert_pp_to_txt import convert_pp_to_txt

from main import batchstart
from main import batchend
from main import fullautocut
from main import sendmail
from main import actupdf


class FileNotFoundException(Exception):
    pass

class PdfThread(QThread):
    calculation_finished = pyqtSignal(str)  # Signal to be emitted when calculation is finished

    def __init__(self,prePdfdir ,infodir,parent=None):
        super().__init__(parent)
        self.prePdfdir = prePdfdir
        self.infodir = infodir

    def run(self):
        # intensive calculation here
        p = actupdf(self.prePdfdir,self.infodir)
        # After the calculation, emit the finished signal
        self.calculation_finished.emit(p)

class SendMailThread(QThread):
    calculation_finished = pyqtSignal(str)  # Signal to be emitted when calculation is finished

    def __init__(self, configdir, maildir,prepdfdir, parent=None):
        super().__init__(parent)
        self.configdir = configdir
        self.maildir = maildir
        self.prepdfdir=prepdfdir

    def run(self):
        # intensive calculation here
        p = sendmail(self.configdir,self.maildir,self.prepdfdir)
        # After the calculation, emit the finished signal
        self.calculation_finished.emit(p)

class AutoCutThread(QThread):
    calculation_finished = pyqtSignal(list)  # Signal to be emitted when calculation is finished

    def __init__(self, supdirs,points,debug, parent=None):
        super().__init__(parent)
        self.supdirs = supdirs
        self.points = points
        self.debug = debug

    def run(self):
        # intensive calculation here
        p = fullautocut(self.supdirs,self.points,self.debug)
        # After the calculation, emit the finished signal
        self.calculation_finished.emit(p)
        
        
class FirstCalculationThread(QThread):
    calculation_finished = pyqtSignal(str)  # Signal to be emitted when calculation is finished

    def __init__(self, dirs, noise,objectkernel,deformkernel,numref,parent=None):
        super().__init__(parent)
        self.dirs = dirs
        self.noise = noise
        self.objectkernel = objectkernel
        self.deformkernel = deformkernel
        self.numref = numref

    def run(self):
        # intensive calculation here
        p = batchstart(self.dirs,self.noise,self.objectkernel,self.deformkernel,self.numref)
        # After the calculation, emit the finished signal
        self.calculation_finished.emit(p)
        

class SecondCalculationThread(QThread):
    calculation_finished = pyqtSignal(str)  # Signal to be emitted when calculation is finished
    
    def __init__(self, dzipfile, refnum,infodir,parent=None):
        super().__init__(parent)
        self.dzipfile = dzipfile
        self.refnum = refnum
        self.infodir = infodir
    
    def run(self):
        # intensive calculation heretttw
        p = batchend(self.dzipfile,self.refnum,self.infodir)
        # After the calculation, emit the finished signal
        self.calculation_finished.emit(p)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.surfdirs = []
        self.zipdir = ""
        self.refnum = []
        self.supdirs = []
        self.problemdirs = []
        
        
        #Valeurs pour lancer l'autocut
        self.verif = True
        self.widget_activated = False
        self.defining_up_vector = False
        self.pick_ref_point = False
        self.closed = False
        self.selected_points = []
        self.temppoints = [] #pour le retry
        self.accepteddirs = []
        
        self.points = []
        self.upnormals = []
        self.refpoints = []
        
        
        self.dzipfile = ""
        self.infodir = ""
        self.configdir = ""
        self.maildir = ""
        
        
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./ut3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(733, 728)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(530, 0, 191, 181))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.noise = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.noise.setSingleStep(0.01)
        self.noise.setProperty("value", 0.1)
        self.noise.setObjectName("noise")
        self.verticalLayout.addWidget(self.noise)
        self.objectkernel = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.objectkernel.setSingleStep(0.1)
        self.objectkernel.setProperty("value", 1.0)
        self.objectkernel.setObjectName("objectkernel")
        self.verticalLayout.addWidget(self.objectkernel)
        self.deformkernel = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.deformkernel.setSingleStep(0.1)
        self.deformkernel.setProperty("value", 1.0)
        self.deformkernel.setObjectName("deformkernel")
        self.verticalLayout.addWidget(self.deformkernel)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 30, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(470, 70, 51, 51))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(440, 120, 91, 61))
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.aligtab = QtWidgets.QWidget()
        self.aligtab.setObjectName("aligtab")
        self.cuttab = QtWidgets.QWidget()
        self.cuttab.setObjectName("cuttab")
        self.label_9 = QtWidgets.QLabel(self.cuttab)
        self.label_9.setGeometry(QtCore.QRect(10, 0, 151, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.cuttab)
        self.label_10.setGeometry(QtCore.QRect(210, 10, 71, 16))
        self.label_10.setObjectName("label_10")
        self.autocutselect = QtWidgets.QPushButton(self.cuttab)
        self.autocutselect.setGeometry(QtCore.QRect(40, 30, 141, 51))
        self.autocutselect.setObjectName("autocutselect")
        self.selectiontype2 = QtWidgets.QComboBox(self.cuttab)
        self.selectiontype2.setGeometry(QtCore.QRect(210, 30, 81, 51))
        self.selectiontype2.setObjectName("selectiontype2")
        self.selectiontype2.addItem("")
        self.selectiontype2.addItem("")
        self.debugtype = QtWidgets.QComboBox(self.cuttab)
        self.debugtype.setGeometry(QtCore.QRect(100, 165, 90, 40))
        self.debugtype.setObjectName("debugtype")
        self.debugtype.addItem("")
        self.debugtype.addItem("")
        self.debugtype.addItem("")
        self.startautocut = QtWidgets.QPushButton(self.cuttab)
        self.startautocut.setGeometry(QtCore.QRect(20, 130, 191, 31))
        self.startautocut.setObjectName("startautocut")
        self.startautocut.setEnabled(False)
        self.opencutdir = QtWidgets.QPushButton(self.cuttab)
        self.opencutdir.setEnabled(False)
        self.opencutdir.setGeometry(QtCore.QRect(230, 120, 81, 51))
        self.opencutdir.setFlat(False)
        self.opencutdir.setObjectName("opencutdir")
        self.thenalign = QtWidgets.QCheckBox(self.cuttab)
        self.thenalign.setGeometry(QtCore.QRect(10, 100, 91, 21))
        self.thenalign.setCheckable(True)
        self.thenalign.setChecked(True)
        self.thenalign.setObjectName("thenalign")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 371, 261))
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.addTab(self.cuttab, "")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.alignementselec = QtWidgets.QPushButton(self.aligtab)
        self.alignementselec.setGeometry(QtCore.QRect(10, 20, 141, 51))
        self.alignementselec.setObjectName("alignementselec")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 290, 721, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(380, 0, 20, 291))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_4 = QtWidgets.QLabel(self.aligtab)
        self.label_4.setGeometry(QtCore.QRect(10, 0, 151, 21))
        self.label_4.setObjectName("label_4")
        self.selectiontype = QtWidgets.QComboBox(self.aligtab)
        self.selectiontype.setGeometry(QtCore.QRect(180, 30, 81, 41))
        self.selectiontype.setObjectName("selectiontype")
        self.selectiontype.addItem("")
        self.selectiontype.addItem("")
        self.clearbox = QtWidgets.QCheckBox(self.aligtab)
        self.clearbox.setGeometry(QtCore.QRect(20, 80, 101, 31))
        self.clearbox.setCheckable(True)
        self.clearbox.setChecked(False)
        self.clearbox.setObjectName("clearbox")
        self.writeply = QtWidgets.QCheckBox(self.aligtab)
        self.writeply.setGeometry(QtCore.QRect(20, 110, 111, 41))
        self.writeply.setCheckable(False)
        self.writeply.setChecked(False)
        self.writeply.setObjectName("writeply")
        self.start1 = QtWidgets.QPushButton(self.aligtab)
        self.start1.setGeometry(QtCore.QRect(10, 150, 241, 71))
        self.start1.setObjectName("start1")
        self.start1.setEnabled(False)
        self.refsurfselect = QtWidgets.QSpinBox(self.centralwidget)
        self.refsurfselect.setGeometry(QtCore.QRect(570, 200, 101, 24))
        self.refsurfselect.setProperty("value", 1)
        self.refsurfselect.setObjectName("refsurfselect")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(440, 200, 131, 21))
        self.label_5.setObjectName("label_5")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(190, 510, 351, 41))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(50, 320, 81, 16))
        self.label_6.setObjectName("label_6")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(0, 560, 731, 121))
        self.listWidget.setObjectName("listWidget")
        self.resSelect = QtWidgets.QPushButton(self.centralwidget)
        self.resSelect.setGeometry(QtCore.QRect(20, 340, 141, 51))
        self.resSelect.setObjectName("resSelect")
        self.start2 = QtWidgets.QPushButton(self.centralwidget)
        self.start2.setGeometry(QtCore.QRect(20, 400, 141, 41))
        self.start2.setObjectName("start2")
        self.openzipdir = QtWidgets.QPushButton(self.aligtab)
        self.openzipdir.setEnabled(False)
        self.openzipdir.setGeometry(QtCore.QRect(270, 150, 81, 71))
        self.openzipdir.setFlat(False)
        self.openzipdir.setObjectName("openzipdir")
        self.openzipdir.clicked.connect(self.open_file_explorer)
        self.label_7 = QtWidgets.QLabel(self.aligtab)
        self.label_7.setGeometry(QtCore.QRect(180, 10, 71, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.aligtab)
        self.label_8.setGeometry(QtCore.QRect(270, 10, 81, 16))
        self.label_8.setObjectName("label_8")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(320, 320, 131, 16))
        self.label_12.setObjectName("label_12")
        self.SENDMAIL = QtWidgets.QPushButton(self.centralwidget)
        self.SENDMAIL.setGeometry(QtCore.QRect(310, 410, 171, 31))
        self.SENDMAIL.setObjectName("SENDMAIL")
        self.configSelec = QtWidgets.QPushButton(self.centralwidget)
        self.configSelec.setGeometry(QtCore.QRect(310, 340, 81, 61))
        self.configSelec.setObjectName("configSelec")
        self.pdfactu = QtWidgets.QPushButton(self.centralwidget)
        self.pdfactu.setGeometry(QtCore.QRect(180, 340, 111, 81))
        self.pdfactu.setObjectName("pdfactu")
        self.mailListSelec = QtWidgets.QPushButton(self.centralwidget)
        self.mailListSelec.setGeometry(QtCore.QRect(400, 340, 81, 61))
        self.mailListSelec.setObjectName("mailListSelec")
        self.zipacces = QtWidgets.QLabel(self.centralwidget)
        self.zipacces.setGeometry(QtCore.QRect(0, 480, 731, 31))
        self.zipacces.setText("")
        self.zipacces.setObjectName("zipacces")
        self.decouptype = QtWidgets.QComboBox(self.aligtab)
        self.decouptype.setGeometry(QtCore.QRect(270, 30, 81, 41))
        self.decouptype.setObjectName("decouptype")
        self.decouptype.addItem("")
        self.tabWidget.addTab(self.aligtab, "")
        self.tabWidget.addTab(self.cuttab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 733, 20))
        self.menubar.setObjectName("menubar")
        self.fileMenu = self.menubar.addMenu("Fichier")
        
        saveAction = QtWidgets.QAction('Sauvegarder', MainWindow)
        saveAction.setShortcut(QtGui.QKeySequence.Save)
        saveAction.triggered.connect(self.save)
        
        openAction = QtWidgets.QAction('Ouvrir', MainWindow)
        openAction.setShortcut(QtGui.QKeySequence.Open)
        openAction.triggered.connect(self.charge)

        self.fileMenu.addAction(saveAction)
        self.fileMenu.addAction(openAction)
        
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.selectiontype.currentIndexChanged.connect(self.selectiontypechanged)
        self.decouptype.currentIndexChanged.connect(self.decouptypechanged) 
        self.alignementselec.clicked.connect(self.select_alig)
        self.autocutselect.clicked.connect(self.select_cut)
        self.resSelect.clicked.connect(self.select_res)
        self.start2.clicked.connect(self.start_res)
        self.start2.setEnabled(False)
        self.start1.clicked.connect(self.start_alig)
        self.startautocut.clicked.connect(self.start_cut)
        self.clearbox.stateChanged.connect(self.updatewply)
        self.configSelec.clicked.connect(self.select_config)
        self.pdfactu.clicked.connect(self.actualize_pdf)
        self.mailListSelec.clicked.connect(self.select_mail)
        self.SENDMAIL.clicked.connect(self.send_mail)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.tabWidget.setCurrentIndex(1)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Dental Notation"))
        self.label.setText(_translate("MainWindow", "noise"))
        self.label_2.setText(_translate("MainWindow", "object kernel width"))
        self.label_3.setText(_translate("MainWindow", "deformation kernel width"))
        self.alignementselec.setText(_translate("MainWindow", "Select"))
        self.label_4.setText(_translate("MainWindow", "Alignement"))
        self.selectiontype.setItemText(0, _translate("MainWindow", "manual"))
        self.selectiontype.setItemText(1, _translate("MainWindow", "auto"))
        self.writeply.setText(_translate("MainWindow", "Write .ply"))
        self.start1.setText(_translate("MainWindow", "Launch"))
        self.label_5.setText(_translate("MainWindow", "Reference surface"))
        self.label_6.setText(_translate("MainWindow", "Postprocess"))
        self.resSelect.setText(_translate("MainWindow", "Select"))
        self.start2.setText(_translate("MainWindow", "Launch"))
        self.openzipdir.setText(_translate("MainWindow", "Open File"))
        self.label_7.setText(_translate("MainWindow", "Selection"))
        self.label_8.setText(_translate("MainWindow", "Cutting"))
        self.decouptype.setItemText(0, _translate("MainWindow", "manuel"))
        self.clearbox.setText(_translate("MainWindow", "clear"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.aligtab), _translate("MainWindow", "Alignement"))
        self.label_9.setText(_translate("MainWindow", "Autocut"))
        self.label_10.setText(_translate("MainWindow", "Selection"))
        self.autocutselect.setText(_translate("MainWindow", "Select"))
        self.selectiontype2.setItemText(0, _translate("MainWindow", "manuel"))
        self.selectiontype2.setItemText(1, _translate("MainWindow", "auto"))
        self.debugtype.setItemText(0, _translate("MainWindow", "no debug"))
        self.debugtype.setItemText(1, _translate("MainWindow", "debug"))
        self.debugtype.setItemText(2, _translate("MainWindow", "hard debug"))
        self.startautocut.setText(_translate("MainWindow", "Launch"))
        self.opencutdir.setText(_translate("MainWindow", "Open File"))
        self.thenalign.setText(_translate("MainWindow", "Align"))
        self.configSelec.setText(_translate("MainWindow", "Config"))
        self.pdfactu.setText(_translate("MainWindow", "Actualize PDF"))
        self.mailListSelec.setText(_translate("MainWindow", "Mails liste"))
        self.label_12.setText(_translate("MainWindow", "Mail Sending"))
        self.SENDMAIL.setText(_translate("MainWindow", "SEND"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.cuttab), _translate("MainWindow", "Autocut"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.cuttab), _translate("MainWindow", "Autocut"))
      
    def updatewply(self):
        if self.clearbox.isChecked():
            self.writeply.setCheckable(True)
        else:
            self.writeply.setCheckable(False)
        
    def selectiontypechanged(self):
        self.resetstart1()
    
    def decouptypechanged(self):
        self.resetstart1()
    
    def resetstart1(self):
        self.surfdirs = []
        self.refnum = []
        self.zipdir = ""
        self.start1.setEnabled(False)
        self.openzipdir.setEnabled(False)
        
    def select_res(self):
        self.dzipfile, _ = QFileDialog.getOpenFileName(
            None, "Select the archive", "~/", "Zip Archive (*.zip)"
        )
        if self.dzipfile is not None and self.dzipfile:
            self.infodir, _ = QFileDialog.getOpenFileName(
                None, "Select the directory containing the students informations", path.dirname(self.dzipfile), "Ods file (*.ods) ;; Csv file (*.csv)"
            )
        if self.dzipfile is not None and (self.dzipfile and len(self.surfdirs) > 0) and self.infodir is not None and self.infodir:
            self.start2.setEnabled(True)
            print("Voici le fichier selectionné : " + str(self.dzipfile))
        else:
            self.start2.setEnabled(False)
            self.display_error("Tout les fichiers n'ont pas été sélectionnés")
            print(self.dzipfile)
            print("Pas de fichier sélectionné")
            
        
    def start_res(self):
        if hasattr(self, 'second_calculation_thread') and self.second_calculation_thread.isRunning():
            # Optionally, handle the case where the thread is already running
            print("Déjà en cours")
            return
        self.second_calculation_thread = SecondCalculationThread(self.dzipfile,self.refnum,infodir=self.infodir)
        self.second_calculation_thread.calculation_finished.connect(self.on_second_calculation_finished)
        self.second_calculation_thread.start()
    
    def show_pyvista_window(self,mdir=None,mesh=None):
        self.verif = True
        self.closed = False
        self.plotter = CBPlotter(self)
        if mdir:
            try:
                self.mesh = pv.read(mdir)
                self.plotter.add_mesh(self.mesh, show_edges=False, color="lightblue")
            except:
                print("Error reading the mesh file")
        else:
            try:
                self.actor = self.plotter.add_mesh(mesh, show_edges=False, color="lightblue")
            except:
                print("Error loading the mesh")
        # Activer la sélection des points
        #self.plotter.add_orientation_widget(actor=self.mesh,interactive=True)
        if self.defining_up_vector:
            self.plotter.add_key_event("a", self.toggle_plane_activation)
            self.planewidget = self.plotter.add_plane_widget(callback= lambda x: (),interaction_event='always')
        elif self.pick_ref_point:
            self.plotter.add_key_event("a", self.toggle_sphere_activation)
            self.spwidget = self.plotter.add_sphere_widget(callback= lambda x: (),interaction_event='always',radius=3,color='red',)
        else:
            #self.plotter.enable_point_picking(callback=self.picked_callback,show_point=True, color="red", point_size=10,pickable_window=True)
            self.plotter.add_key_event("b", lambda : ())
            self.plotter.enable_surface_picking(callback=self.picked_callback,show_point=True, color="red", point_size=20,pickable_window=True)
    
    
    def picked_callback(self, point_coord):
    
        self.selected_points.append(point_coord)
        print(f"Point sélectionné : {point_coord}")

        if not self.defining_up_vector:
            # Si trois points sont sélectionnés, fermez la fenêtre
            if len(self.selected_points) == 3:
                # Réinitialisez pour la définition du "haut"
                self.points.append(tuple(self.selected_points))
                self.selected_points = []
                self.close_pyvista_window()
                self.defining_up_vector = True
                self.show_pyvista_window(mesh=self.mesh)
                
    def toggle_sphere_activation(self):
        self.widget_activated = True
        self.defineRef(self.spwidget.GetCenter())
    
    def toggle_plane_activation(self):
        self.widget_activated = True
        self.defineUp(self.planewidget.GetNormal(),self.planewidget.GetOrigin())
    
    def defineUp(self,normal,origin):
        if not self.plotter: 
            print("Plotter is closed, ignoring callback")
            return
        if self.widget_activated:
            print("La normale est : " + str(normal))
            self.defining_up_vector = False
            self.selected_points = []
            self.upnormals.append(normal)
            self.widget_activated = False
            self.closed = True
            self.close_pyvista_window()
            self.pick_ref_point = True
            self.show_pyvista_window(mesh=self.mesh)

    def defineRef(self,point):
        if not self.plotter: 
            print("Plotter is closed, ignoring callback")
            return
        if self.widget_activated:
            print("Le point de référence est : " + str(point))
            self.pick_ref_point = False
            self.selected_points = []
            self.mesh = None
            self.refpoints.append(point)
            self.widget_activated = False
            self.closed = True
            self.close_pyvista_window(finish=True)
            

    def close_pyvista_window(self,finish=False):
        
        if finish:
            self.verif = True
            self.mesh = None
        else:
            self.verif = False
        if self.plotter is not None:
            try:
                self.plotter.reset_key_events()
                self.plotter.clear_plane_widgets()
                self.plotter.close()
            except Exception as e:
                print("Erreur lors de la fermeture du plotter:", e)
            self.plotter = None
            
    def select_points_sup(self,dirs):
        points = []
        self.problemdirs = []
        self.points = []
        self.upnormals = []
        self.refpoints = []
        self.verif = True
        self.widget_activated = False
        self.defining_up_vector = False
        self.pick_ref_point = False
        self.closed = False
        for d in dirs:
            self.show_pyvista_window(d)
            while (not self.closed):  # La boucle d'attente
                QApplication.processEvents()
            if (len(self.points) == 0):
                print("Pas de points sélectionnés")
                self.problemdirs.append((d,"Pas de points sélectionnés"))
            elif len(self.points[-1]) < 3:
                print("Problème de sélection des points, le dernier elt de points est :" + str(self.points[-1]))
                self.points.pop()
                self.problemdirs.append((d,"Problème de sélection des points"))
            elif len(self.points) != len(self.upnormals):
                print("Normale pas selectionnée")
                self.points.pop()
                self.problemdirs.append((d,"Normale pas selectionnée"))
            elif len(self.refpoints) != len(self.upnormals):
                print("Point de référence pas selectionné")
                self.points.pop()
                self.upnormals.pop()
                self.problemdirs.append((d,"Point de référence pas selectionné"))
            else:
                self.accepteddirs.append(d)
            
            self.closed = False
        points = list(zip(self.points,self.upnormals,self.refpoints))
        #gérer le fait que puisque on propose de retry sur des dirs, dirs et points seront pas forcément dans le même ordre + gérer pour renvoyer les points où il faut
        if len(self.problemdirs) > 0:
            pcorrec = self.showErrorDialog(self.problemdirs,self.select_points_sup)
            print("caca")
            points.extend(pcorrec)
        self.temppoints = points
        return points
    
    def send_mail(self):
        if hasattr(self, 'mail_thread') and self.mail_thread.isRunning():
            # Optionally, handle the case where the thread is already running
            print("Déjà en cours")
            return
        if self.configdir is None:
            self.select_config()
        if self.maildir is None:
            self.select_mail()
        if not self.configdir or not self.maildir:
            print("Pas de config ou de mail")
            self.display_error("Pas de config ou de mail")
            return
        self.mail_thread = SendMailThread(self.configdir,self.maildir,self.dzipfile)
        self.mail_thread.calculation_finished.connect(self.on_mail_finished)
        self.mail_thread.start()
    
    def start_cut(self):
        self.accepteddirs = []
        #self.show_pyvista_window(self.supdirs[0])
        points = self.select_points_sup(self.supdirs)
        self.supdirs = self.accepteddirs
        print("les points : " + str(points))
        print("Les dirs dans leur nouvel ordre : " + str(self.supdirs))
        if hasattr(self, 'autocut_thread') and self.autocut_thread.isRunning():
            # Optionally, handle the case where the thread is already running
            print("Déjà en cours")
            return
        self.autocut_thread = AutoCutThread(self.supdirs,points,debug=self.debugtype.currentIndex())
        self.autocut_thread.calculation_finished.connect(self.on_autocut_finished)
        self.autocut_thread.start()
    
    def actualize_pdf(self):
        if hasattr(self, 'pdf_thread') and self.pdf_thread.isRunning():
            # Optionally, handle the case where the thread is already running
            print("Déjà en cours")
            return
        if self.infodir is None or not self.infodir or self.infodir == "":
            self.display_error("Pas de fichier d'informations, il faut d'abord run postprocess (ou charger une sauvegarde)")
            return
        if self.dzipfile is None or not self.dzipfile or self.dzipfile == "":
            self.display_error("Pas de fichier zip, il faut d'abord run postprocess (ou charger une sauvegarde)")
            return
        self.pdf_thread = PdfThread(infodir= self.infodir,prePdfdir = self.dzipfile)
        self.pdf_thread.calculation_finished.connect(self.on_pdf_finished)
        self.pdf_thread.start()
    
    def start_alig(self):
        if hasattr(self, 'calculation_thread') and self.calculation_thread.isRunning():
            # Optionally, handle the case where the thread is already running
            print("Déjà en cours")
            return
        self.calculation_thread = FirstCalculationThread(self.surfdirs,self.noise.value(),self.objectkernel.value(),self.deformkernel.value(),self.refnum)
        self.calculation_thread.calculation_finished.connect(self.on_calculation_finished)
        self.calculation_thread.start()

    def on_pdf_finished(self,path):
        #self.display_info("PDF actualisé")
        pass

    def on_autocut_finished(self,paths):
        for pointspic,surface,surfacecut in paths:
            self.surfdirs.append((pointspic,surface,surfacecut))
            self.update_surf_list(pointspic,surface,surfacecut)
        self.refnum += [self.refsurfselect.value()]*len(self.surfdirs)
        self.start1.setEnabled(True)
        if self.thenalign.isChecked() and self.surfdirs is not None and (self.surfdirs and len(self.surfdirs) > 0):
            self.start_alig()
        
    def on_calculation_finished(self,path):
        self.zipdir = path
        self.openzipdir.setEnabled(True)
        self.zipacces.setText("Accès à l'archive : " + path)
        self.save()
        # Update the UI with the results of the calculation
        
    def on_second_calculation_finished(self,path):
        pass
    
    def select_alig(self):
        if self.selectiontype.currentText() == "manuel":
            self.manu_user_select_files()
        if self.selectiontype.currentText() == "auto":
            self.auto_user_select_files()
        self.on_file_selected()
        
    def select_cut(self):
        if self.selectiontype2.currentText() == "manuel":
            self.manu_user_select_cut()
        if self.selectiontype2.currentText() == "auto":
            self.auto_user_select_cut()
        self.on_file_selected()

    def on_file_selected(self):
        if self.surfdirs is not None and (self.surfdirs and len(self.surfdirs) > 0):
            print(self.surfdirs)
            self.start1.setEnabled(True)
        else :
            self.start1.setEnabled(False)
            print(self.surfdirs)
            print("Pas de fichier sélectionné")
        if self.supdirs is not None and (self.supdirs and len(self.supdirs) > 0):
            print(self.supdirs)
            self.startautocut.setEnabled(True)
        else :
            self.startautocut.setEnabled(False)
            print(self.supdirs)
            print("Pas de fichier sélectionné")

    
    def user_select_files(self):
        pointsFile, _ = QFileDialog.getOpenFileName(
            None, "Select the txt file containing points", "~/", "Text Files (*.txt)"
        )
        if not pointsFile:
            raise FileNotFoundException("No points file selected")
        
        surfaceFile, _ = QFileDialog.getOpenFileName(
            None, "Select the surface", path.dirname(pointsFile), "PLY Files (*.ply)"
        )
        
        if not surfaceFile:
            raise FileNotFoundException("No surface file selected")

        surfaceFileCut, _ = QFileDialog.getOpenFileName(
            None, "Select the cut surfaces", path.dirname(pointsFile), "PLY Files (*.ply)"
        )
        
        if not surfaceFileCut:
            raise FileNotFoundException("No surface cut file selected")

        return pointsFile, surfaceFile, surfaceFileCut

    def user_select_cut(self):
        supFile, _ = QFileDialog.getOpenFileName(
            None, "Select the stl file", "~/", "STL Files (*.stl)"
        )
        if not supFile:
            raise FileNotFoundException("No support file selected")
        return supFile

    def update_surf_list(self,pointFile,surfaceFile,surfaceFileCut):
        self.listWidget.addItems(["POINT FILE : " + str(pointFile) + " SURFACE FILE : "+ str(surfaceFile) + " SURFACE FILE CUT : "+ str(surfaceFileCut)])

    def update_sup_list(self,supFile):
        self.listWidget.addItems(["SUPPORT FILE : " + str(supFile)])

    def manu_user_select_files(self):
        try:
            pointsFile, surfaceFile, surfaceFileCut = self.user_select_files()
            self.surfdirs.append((pointsFile, surfaceFile, surfaceFileCut))
            self.refnum.append(self.refsurfselect.value())
            self.update_surf_list(pointsFile,surfaceFile,surfaceFileCut)
        except FileNotFoundException as e:
            print(e)
    
    def manu_user_select_cut(self):
        try:
            supFile = self.user_select_cut()
            self.supdirs.append(supFile)
            self.refnum.append(self.refsurfselect.value())
            self.update_sup_list(supFile)
        except FileNotFoundException as e:
            print(e)

    def auto_user_select_cut(self):
        root_directory = QFileDialog.getExistingDirectory(
            None, "Please select a directory containing your files", "~/"
        )
        
        if root_directory:
            all_files = [f for f in os.listdir(root_directory) if f.endswith(".stl")]
            self.supdirs = [os.path.join(root_directory, f) for f in all_files]
            for supFile in self.supdirs:
                self.update_sup_list(supFile)
        else:
            print("Aucun dossier sélectionné")
        self.refnum += [self.refsurfselect.value()]*len(self.surfdirs)

    def auto_user_select_files(self):
        root_directory = QFileDialog.getExistingDirectory(
            None, "Please select a directory containing your files", "~/"
        )

        if root_directory:
                # Recueillir tous les fichiers
            all_files = [f for f in os.listdir(root_directory) if f.endswith(".ply") or f.endswith(".txt")]

            # Crée un dictionnaire avec une liste vide comme valeur par défaut pour chaque clé
            files_by_prefix = defaultdict(list)

            # Identifier les fichiers .ply qui ne contiennent pas "cut" dans leur nom comme préfixes
            for filename in all_files:
                if filename.endswith(".ply") and "cut" not in filename:
                    prefix = re.split('\.ply', filename)[0]
                    files_by_prefix[prefix].append(os.path.join(root_directory, filename))

            # Chercher les autres fichiers correspondant à ces préfixes
            for filename in all_files:
                if filename.endswith(".txt") or "cut" in filename:
                    for prefix in files_by_prefix.keys():
                        if filename.startswith(prefix):
                            files_by_prefix[prefix].append(os.path.join(root_directory, filename))
                            break
                elif filename.endswith(".pp"):
                    for prefix in files_by_prefix.keys():
                        if filename.startswith(prefix):
                            # Vérifier si le fichier .txt correspondant existe déjà
                            txt_file = os.path.join(root_directory, f"{prefix}.txt")
                            if txt_file not in files_by_prefix[prefix]:
                                input_file = os.path.join(root_directory, filename)
                                output_file = txt_file
                                convert_pp_to_txt(input_file, output_file)
                                files_by_prefix[prefix].append(output_file)
                            break

            triplets = []
            for prefix, files in files_by_prefix.items():
                if len(files) == 3:
                    # Trie les fichiers pour toujours avoir l'ordre suivant : .txt, .ply, _cut.ply
                    files.sort(key=lambda x: (not x.endswith(".txt"), not x.endswith(".ply"), "cut" in x))
                    triplets.append(tuple(files))
                    self.update_surf_list(files[0],files[1],files[2])

            self.surfdirs = triplets
            self.refnum += [self.refsurfselect.value()]*len(self.surfdirs)
        else:
            print("Aucun dossier sélectionné")
    
    def select_config(self):
        
        cffile,_ = QFileDialog.getOpenFileName(
            None, "Select the config file", "~/", "Config Files (*.cfg)"
        )
        if not cffile:
            self.display_error("No config file was selected. Please try again.")
        else:
            self.configdir = cffile
            
    def select_mail(self):
        mailfile,_ = QFileDialog.getOpenFileName(
            None, "Select the mail file", "~/", "Mail Files (*.csv)"
        )
        if not mailfile:
            self.display_error("No mail file was selected. Please try again.")
        else:
            self.maildir = mailfile
    
    def open_file_explorer(self):
        print("open : " + self.zipdir)
        print("platform : " + sys.platform)
        if os.name == 'nt':  # Pour Windows
            subprocess.Popen(['explorer', self.zipdir])
        elif os.name == 'posix':  # Pour macOS
            subprocess.Popen(['open', self.zipdir])
        else:
            print("OS non supporté")
    
    def save(self):
        
        filenameprop = f'save_{str(uuid.uuid4())[:8]}.pkl'
        
        options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Enregistrer la sauvegarde",filenameprop,"Pickle Files (*.pkl)", options=options)
         
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump({'refnum': self.refnum, 'dirs': self.surfdirs, 'config' : self.configdir,'supdirs' : self.supdirs, 'maildir' : self.maildir,'dzipfile' : self.dzipfile, 'infodir' : self.infodir }, f) 
    
    def charge(self):
        fname = QFileDialog.getOpenFileName(None, 'Ouvrir fichier', '/', "Pickle files (*.pkl)")

        if fname[0]:
            with open(fname[0], 'rb') as f:
                data = pickle.load(f)
            self.refnum = data['refnum']
            self.surfdirs = data['dirs']
            self.configdir = data['config']
            self.supdirs = data['supdirs']
            self.maildir = data['maildir']
            if 'infodir' in data:
                self.infodir = data['infodir']
            if 'dzipfile' in data:
                self.dzipfile = data['dzipfile']
            self.update_ui()
            
    def update_ui(self):
        self.listWidget.clear()
        for j in range(len(self.supdirs)):
            self.update_sup_list(self.supdirs[j])
        for i in range(len(self.surfdirs)):
            self.update_surf_list(self.surfdirs[i][0],self.surfdirs[i][1],self.surfdirs[i][2])
        if self.refnum is not None and (self.refnum and len(self.refnum) > 0):
            self.refsurfselect.setValue(self.refnum[0])
        if self.dzipfile is not None and (self.dzipfile and len(self.surfdirs) > 0) and self.infodir is not None and self.infodir:
            self.start2.setEnabled(True)
            print("Voici le fichier selectionné : " + str(self.dzipfile))
        self.on_file_selected()
        #self.openzipdir.setEnabled(True)

    def display_error(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(message)
        msgBox.setWindowTitle("Error")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
        
    def showErrorDialog(self,dir_info,retry_function):
        self.temppoints = []
        dialog = QDialog()
        dialog.setWindowTitle("Erreur")
        
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        
        # Construire le texte à afficher
        text = "Les répertoires suivants ont posé problème :\n"
        for directory, info in dir_info:
            text += f" - {directory} : {info}\n"
            
        text_edit.setPlainText(text)
        
        layout.addWidget(text_edit)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        retry_button = QPushButton("Réessayer")
        retry_button.clicked.connect(lambda: retry_function([d[0] for d in dir_info]))
        layout.addWidget(retry_button)
        
        dialog.setLayout(layout)
        dialog.exec()
        
        return self.temppoints
        
        
    

class CBPlotter(BackgroundPlotter):
    def __init__(self,mw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mw = mw
        
    
    def closeEvent(self, event):
        print("La fenêtre est en train de se fermer")
        self.mw.widget_activated = False
        self.mw.defining_up_vector = False
        self.mw.selected_points = []
        self.mw.pick_ref_point = False
        if self.mw.verif:
            self.mw.closed = True
        else:
            self.mw.closed = False
        
        super().closeEvent(event)  # Le close parent




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
