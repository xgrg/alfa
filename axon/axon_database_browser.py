#!/usr/bin/env python

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Axon database browser')
    opts = parser.parse_args()

    from brainvisa import axon, processes

    axon.initializeProcesses()
    from brainvisa.data.qtgui.hierarchyBrowser import HierarchyBrowser
    from PyQt4 import QtGui, QtCore, Qt
    import sys

    app = Qt.QApplication( sys.argv )
    a = HierarchyBrowser()
    a.show()
    app.exec_()

