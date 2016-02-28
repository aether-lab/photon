# -------------------------------------------------
# Project created by QtCreator 2010-04-19T22:10:23
# -------------------------------------------------


DEFINES += SCHLIEREN_LIBRARY
SOURCES += main.cpp \
    schlierenrenderer.cpp \
    schlierenimagefilter.cpp \
    schlierenfilter.cpp
HEADERS += opengl_include.h \
    cutil.h \
    cutil_inline.h \
    cutil_math.h \
    cutil_gl_inline.h \
    cutil_inline_runtime.h \
    main.h \
    kernel_volume.h \
    kernel_post.h \
    kernel_functions.cu \
    schlierenrenderer.h \
    schlierenimagefilter.h \
    schlierenfilter.h \
    RenderParameters.h \
    kernel_render.h \
    kernel_functions.h \
    kernel_cutoff.h \
    kernel_filter.h
OTHER_FILES += host.cu \
    host_render.cu \
    CMakeLists.txt

INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/local/cuda/samples/common/inc
LIBS += -L/usr/local/lib -lteem
LIBS += -L/usr/local/cuda/lib64
