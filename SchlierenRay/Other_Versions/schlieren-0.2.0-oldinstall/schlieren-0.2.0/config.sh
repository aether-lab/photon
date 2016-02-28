PKG_VERSION=0.2.0
PKG_NAME=schlieren
PKG_DESCRIPTORS=
REQUIRES_IN_SRC_BUILD=false
OPTIONS=

PKG_ID="$PKG_NAME-$PKG_VERSION"
SRC_DIR="$PKG_ID-src"
BUILD_DIR="$PKG_ID-Build"
SCRIPT="$(pwd)/config.sh"

if [ ! -z $PKG_DESCRIPTORS ]; then
    INSTALL_DIR="$PKG_ID-$PKG_DESCRIPTORS"
else
    INSTALL_DIR="$PKG_ID"
fi

if [ -d $INSTALL_DIR ]; then
    rm -rf $INSTALL_DIR
fi
if [ -d $BUILD_DIR ]; then
    rm -rf $BUILD_DIR
fi
mkdir $INSTALL_DIR
mkdir $BUILD_DIR

SRC_DIR="$(pwd)/$SRC_DIR"
INSTALL_DIR="$(pwd)/$INSTALL_DIR"

if [ $REQUIRES_IN_SRC_BUILD == true ]; then
    cp -rf "$SRC_DIR/." $BUILD_DIR
    SRC_DIR="$(pwd)/$BUILD_DIR"
fi

cd $BUILD_DIR

cmake $SRC_DIR -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
 -DCMAKE_BUILD_TYPE:STRING=Debug \
 -DCUDA_SDK_ROOT_DIR:PATH=/usr/local/cuda \
 -DFOUND_TEEM_BIN:PATH=/usr/local/bin \
 -DFOUND_TEEMCONFIG_CMAKE:PATH=/usr/local/lib \
 -DTEEM_INCLUDE_DIRS:PATH=/usr/local/include \
 -DCMAKE_CXX_FLAGS:STRING="-I /usr/local/cuda/samples/common/inc -I $SRC_DIR"
make &> make.txt
#make install

mkdir $INSTALL_DIR/bin
mkdir $INSTALL_DIR/lib
mkdir $INSTALL_DIR/include
cp -f $SCRIPT $INSTALL_DIR/config.sh
cp -f ./schlieren $INSTALL_DIR/bin/
cp -f ./libSchlieren.so $INSTALL_DIR/lib/
cp -f $SRC_DIR/schlierenfilter.h $INSTALL_DIR/include/
cp -f $SRC_DIR/schlierenimagefilter.h $INSTALL_DIR/include/
cp -f $SRC_DIR/schlierenrenderer.h $INSTALL_DIR/include/




