/********************************************************************************
** Form generated from reading UI file 'filter.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FILTER_H
#define UI_FILTER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ColorFilter
{
public:

    void setupUi(QWidget *ColorFilter)
    {
        if (ColorFilter->objectName().isEmpty())
            ColorFilter->setObjectName(QStringLiteral("ColorFilter"));
        ColorFilter->resize(290, 283);

        retranslateUi(ColorFilter);

        QMetaObject::connectSlotsByName(ColorFilter);
    } // setupUi

    void retranslateUi(QWidget *ColorFilter)
    {
        ColorFilter->setWindowTitle(QApplication::translate("ColorFilter", "Form", 0));
    } // retranslateUi

};

namespace Ui {
    class ColorFilter: public Ui_ColorFilter {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FILTER_H
