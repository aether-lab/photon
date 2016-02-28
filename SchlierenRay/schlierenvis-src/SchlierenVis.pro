# -------------------------------------------------
# Project created by QtCreator 2010-04-13T12:31:52
# -------------------------------------------------
QT += core gui widgets opengl xml svg
TARGET = SchlierenVis
TEMPLATE = app
SOURCES += main.cpp \
    mainwindow.cpp \
    glview.cpp \
    filter.cpp \
    painterwidget.cpp
HEADERS += mainwindow.h \
    glview.h \
    filter.h \
    painterwidget.h
FORMS += mainwindow.ui \
    filter.ui
INCLUDEPATH += /home/barracuda/a/lrajendr/SchlierenRayVis/SchlierenRay/schlieren-0.2.0-debug/include
INCLUDEPATH += /home/barracuda/a/lrajendr/SchlierenRayVis/SchlierenRay/cuda-samples/NVIDIA_CUDA-7.0_Samples/common/inc
INCLUDEPATH += /home/barracuda/a/lrajendr/SchlierenRayVis/SchlierenRay/cuda-samples/cuda/include/
INCLUDEPATH += /usr/local/include
#LIBS += -LD:/home/carson/svn/Schlieren/build -lSchlieren
#unix:LIBS += -lglut
LIBPATH += /home/barracuda/a/lrajendr/SchlierenRayVis/SchlierenRay/schlieren-0.2.0-debug/bin
LIBPATH += /usr/local/bin
LIBS += -L/home/barracuda/a/lrajendr/SchlierenRayVis/SchlierenRay/schlieren-0.2.0-debug/lib -lSchlieren -lteem
QMAKE_LFLAGS += -L/home/barracuda/a/lrajendr/SchlierenRayVis/SchlierenRay/schlieren-0.2.0-debug/lib
