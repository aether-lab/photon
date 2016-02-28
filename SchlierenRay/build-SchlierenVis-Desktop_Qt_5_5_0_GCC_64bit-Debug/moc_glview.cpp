/****************************************************************************
** Meta object code from reading C++ file 'glview.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../schlierenvis-src/glview.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'glview.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_GLView_t {
    QByteArrayData data[18];
    char stringdata0[160];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GLView_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GLView_t qt_meta_stringdata_GLView = {
    {
QT_MOC_LITERAL(0, 0, 6), // "GLView"
QT_MOC_LITERAL(1, 7, 17), // "drawFilterAtPoint"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 4), // "draw"
QT_MOC_LITERAL(4, 31, 14), // "setImageCutoff"
QT_MOC_LITERAL(5, 46, 6), // "float*"
QT_MOC_LITERAL(6, 53, 3), // "img"
QT_MOC_LITERAL(7, 57, 5), // "width"
QT_MOC_LITERAL(8, 63, 6), // "height"
QT_MOC_LITERAL(9, 70, 8), // "loadData"
QT_MOC_LITERAL(10, 79, 11), // "std::string"
QT_MOC_LITERAL(11, 91, 8), // "filename"
QT_MOC_LITERAL(12, 100, 12), // "setDataScale"
QT_MOC_LITERAL(13, 113, 5), // "scale"
QT_MOC_LITERAL(14, 119, 21), // "setProjectionDistance"
QT_MOC_LITERAL(15, 141, 1), // "d"
QT_MOC_LITERAL(16, 143, 14), // "setCutoffScale"
QT_MOC_LITERAL(17, 158, 1) // "c"

    },
    "GLView\0drawFilterAtPoint\0\0draw\0"
    "setImageCutoff\0float*\0img\0width\0height\0"
    "loadData\0std::string\0filename\0"
    "setDataScale\0scale\0setProjectionDistance\0"
    "d\0setCutoffScale\0c"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GLView[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   49,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    0,   54,    2, 0x0a /* Public */,
       4,    3,   55,    2, 0x0a /* Public */,
       9,    1,   62,    2, 0x0a /* Public */,
      12,    1,   65,    2, 0x0a /* Public */,
      14,    1,   68,    2, 0x0a /* Public */,
      16,    1,   71,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Float, QMetaType::Float,    2,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5, QMetaType::Int, QMetaType::Int,    6,    7,    8,
    QMetaType::Void, 0x80000000 | 10,   11,
    QMetaType::Void, QMetaType::Float,   13,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::Float,   17,

       0        // eod
};

void GLView::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GLView *_t = static_cast<GLView *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->drawFilterAtPoint((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 1: _t->draw(); break;
        case 2: _t->setImageCutoff((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 3: _t->loadData((*reinterpret_cast< std::string(*)>(_a[1]))); break;
        case 4: _t->setDataScale((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 5: _t->setProjectionDistance((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 6: _t->setCutoffScale((*reinterpret_cast< float(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (GLView::*_t)(float , float );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GLView::drawFilterAtPoint)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject GLView::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_GLView.data,
      qt_meta_data_GLView,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *GLView::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GLView::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_GLView.stringdata0))
        return static_cast<void*>(const_cast< GLView*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int GLView::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void GLView::drawFilterAtPoint(float _t1, float _t2)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
