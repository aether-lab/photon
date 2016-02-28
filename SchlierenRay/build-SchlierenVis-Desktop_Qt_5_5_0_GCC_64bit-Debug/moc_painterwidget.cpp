/****************************************************************************
** Meta object code from reading C++ file 'painterwidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../schlierenvis-src/painterwidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'painterwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_PainterWidget_t {
    QByteArrayData data[16];
    char stringdata0[121];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PainterWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PainterWidget_t qt_meta_stringdata_PainterWidget = {
    {
QT_MOC_LITERAL(0, 0, 13), // "PainterWidget"
QT_MOC_LITERAL(1, 14, 12), // "imageChanged"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 6), // "float*"
QT_MOC_LITERAL(4, 35, 3), // "img"
QT_MOC_LITERAL(5, 39, 5), // "width"
QT_MOC_LITERAL(6, 45, 6), // "height"
QT_MOC_LITERAL(7, 52, 12), // "setBrushSize"
QT_MOC_LITERAL(8, 65, 1), // "s"
QT_MOC_LITERAL(9, 67, 13), // "setBrushColor"
QT_MOC_LITERAL(10, 81, 5), // "color"
QT_MOC_LITERAL(11, 87, 9), // "drawBrush"
QT_MOC_LITERAL(12, 97, 1), // "x"
QT_MOC_LITERAL(13, 99, 1), // "y"
QT_MOC_LITERAL(14, 101, 17), // "drawBrushAbsolute"
QT_MOC_LITERAL(15, 119, 1) // "p"

    },
    "PainterWidget\0imageChanged\0\0float*\0"
    "img\0width\0height\0setBrushSize\0s\0"
    "setBrushColor\0color\0drawBrush\0x\0y\0"
    "drawBrushAbsolute\0p"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PainterWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    3,   39,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    1,   46,    2, 0x0a /* Public */,
       9,    1,   49,    2, 0x0a /* Public */,
      11,    2,   52,    2, 0x0a /* Public */,
      14,    1,   57,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, QMetaType::Int, QMetaType::Int,    4,    5,    6,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::QColor,   10,
    QMetaType::Void, QMetaType::Float, QMetaType::Float,   12,   13,
    QMetaType::Void, QMetaType::QPoint,   15,

       0        // eod
};

void PainterWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PainterWidget *_t = static_cast<PainterWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->imageChanged((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 1: _t->setBrushSize((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->setBrushColor((*reinterpret_cast< QColor(*)>(_a[1]))); break;
        case 3: _t->drawBrush((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 4: _t->drawBrushAbsolute((*reinterpret_cast< QPoint(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (PainterWidget::*_t)(float * , int , int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&PainterWidget::imageChanged)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject PainterWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_PainterWidget.data,
      qt_meta_data_PainterWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *PainterWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PainterWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_PainterWidget.stringdata0))
        return static_cast<void*>(const_cast< PainterWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int PainterWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void PainterWidget::imageChanged(float * _t1, int _t2, int _t3)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
