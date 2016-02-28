/****************************************************************************
** Meta object code from reading C++ file 'filter.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../schlierenvis-src/filter.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'filter.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ColorFilterWidget_t {
    QByteArrayData data[20];
    char stringdata[181];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ColorFilterWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ColorFilterWidget_t qt_meta_stringdata_ColorFilterWidget = {
    {
QT_MOC_LITERAL(0, 0, 17),
QT_MOC_LITERAL(1, 18, 12),
QT_MOC_LITERAL(2, 31, 0),
QT_MOC_LITERAL(3, 32, 6),
QT_MOC_LITERAL(4, 39, 3),
QT_MOC_LITERAL(5, 43, 5),
QT_MOC_LITERAL(6, 49, 6),
QT_MOC_LITERAL(7, 56, 9),
QT_MOC_LITERAL(8, 66, 1),
QT_MOC_LITERAL(9, 68, 1),
QT_MOC_LITERAL(10, 70, 11),
QT_MOC_LITERAL(11, 82, 1),
QT_MOC_LITERAL(12, 84, 16),
QT_MOC_LITERAL(13, 101, 18),
QT_MOC_LITERAL(14, 120, 1),
QT_MOC_LITERAL(15, 122, 18),
QT_MOC_LITERAL(16, 141, 17),
QT_MOC_LITERAL(17, 159, 9),
QT_MOC_LITERAL(18, 169, 1),
QT_MOC_LITERAL(19, 171, 9)
    },
    "ColorFilterWidget\0imageChanged\0\0float*\0"
    "img\0width\0height\0drawBrush\0x\0y\0"
    "updateColor\0c\0imageChangedSlot\0"
    "onDataSliderChange\0v\0onProjSliderChange\0"
    "onCutSliderChange\0openImage\0i\0saveImage"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ColorFilterWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    3,   59,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    2,   66,    2, 0x0a /* Public */,
      10,    1,   71,    2, 0x0a /* Public */,
      12,    3,   74,    2, 0x0a /* Public */,
      13,    1,   81,    2, 0x0a /* Public */,
      15,    1,   84,    2, 0x0a /* Public */,
      16,    1,   87,    2, 0x0a /* Public */,
      17,    1,   90,    2, 0x0a /* Public */,
      19,    1,   93,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, QMetaType::Int, QMetaType::Int,    4,    5,    6,

 // slots: parameters
    QMetaType::Void, QMetaType::Float, QMetaType::Float,    8,    9,
    QMetaType::Void, QMetaType::QColor,   11,
    QMetaType::Void, 0x80000000 | 3, QMetaType::Int, QMetaType::Int,    4,    5,    6,
    QMetaType::Void, QMetaType::Int,   14,
    QMetaType::Void, QMetaType::Int,   14,
    QMetaType::Void, QMetaType::Int,   14,
    QMetaType::Void, QMetaType::QString,   18,
    QMetaType::Void, QMetaType::QString,   18,

       0        // eod
};

void ColorFilterWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ColorFilterWidget *_t = static_cast<ColorFilterWidget *>(_o);
        switch (_id) {
        case 0: _t->imageChanged((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 1: _t->drawBrush((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 2: _t->updateColor((*reinterpret_cast< const QColor(*)>(_a[1]))); break;
        case 3: _t->imageChangedSlot((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 4: _t->onDataSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->onProjSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->onCutSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->openImage((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 8: _t->saveImage((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (ColorFilterWidget::*_t)(float * , int , int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&ColorFilterWidget::imageChanged)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject ColorFilterWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_ColorFilterWidget.data,
      qt_meta_data_ColorFilterWidget,  qt_static_metacall, 0, 0}
};


const QMetaObject *ColorFilterWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ColorFilterWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ColorFilterWidget.stringdata))
        return static_cast<void*>(const_cast< ColorFilterWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int ColorFilterWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void ColorFilterWidget::imageChanged(float * _t1, int _t2, int _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
