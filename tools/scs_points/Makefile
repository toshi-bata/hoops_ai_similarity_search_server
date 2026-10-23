# scs_points - Linux (Ubuntu) console build. No Qt, no qmake.
#
# libsc is a plain C/C++ SDK and this tool is a single-file C++17 console
# program, so a small Makefile driving g++ is all that is needed. Point
# HVISUALIZE_WEB_INSTALL_DIR at the root of the HOOPS Visualize Web SDK (the
# folder that contains authoring/libsc) before building:
#
#   cp _linux_env.sh.example _linux_env.sh   # then edit the path
#   source _linux_env.sh                     # exports HVISUALIZE_WEB_INSTALL_DIR
#   make                                     # writes ../../bin/scs_points
#   make clean
#
# Requires g++ >= 9 (for a complete <filesystem>; Ubuntu 22.04/24.04 ship a newer
# one). Only build-essential is needed - no Qt packages.
#
# IMPORTANT: build against the SAME HOOPS Visualize Web version the web viewer
# engine is pinned to (web/package.json: @ts3d-hoops/* = 2026.2.0). A point-cloud
# SCS authored by a NEWER libsc (e.g. 2026.6.0) fails to load in the 2026.2.0
# engine and the client reports "Failed to load the map."
#
# The default lib set below is CONFIRMED against HOOPS Visualize Web 2026.2.0 on
# Ubuntu: the Linux SDK ships STATIC archives (libsc.a, libsc_utils.a,
# libsc_utils_text.a, libz.a) and has NO hps_core library (unlike Windows). The
# bundled library set varies BETWEEN SDK versions (2026.6.0, for instance, also
# ships libpng16.a/libpng.a); override SC_LIBS to match your SDK, for example:
#   make SC_LIBS="-lsc -lsc_utils -lsc_utils_text -lpng16 -lpng -lz -lpthread -ldl -lm"
# Because the archives are static, keep the SDK libraries before the z/png
# dependencies they pull in. Override SC_LIBDIR too if the lib path differs.

SC_ROOT   ?= $(HVISUALIZE_WEB_INSTALL_DIR)/authoring/libsc
SC_INCDIR ?= $(SC_ROOT)/include
SC_LIBDIR ?= $(SC_ROOT)/lib/linux64
SC_LIBS   ?= -lsc -lsc_utils -lsc_utils_text -lz -lpthread -ldl -lm

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

BINDIR = $(CURDIR)/../../bin
TARGET = $(BINDIR)/scs_points

# The SDK libraries are static, so no HOOPS .so is loaded at run time; the
# $ORIGIN rpath only matters if a future SDK ships shared libraries instead.
LDFLAGS += -L$(SC_LIBDIR) -Wl,-rpath,'$$ORIGIN'

.PHONY: all clean
all: $(TARGET)

$(TARGET): main.cpp hoops_license.h
	@test -f "$(SC_INCDIR)/sc_store.h" || { \
	  echo "ERROR: sc_store.h not found under $(SC_INCDIR)."; \
	  echo "       Set HVISUALIZE_WEB_INSTALL_DIR (source _linux_env.sh) or override SC_INCDIR."; \
	  exit 1; }
	@test -f hoops_license.h || { \
	  echo "ERROR: hoops_license.h is missing (download it from the Tech Soft 3D Developer Zone)."; \
	  exit 1; }
	@mkdir -p "$(BINDIR)"
	$(CXX) $(CXXFLAGS) -I"$(SC_INCDIR)" main.cpp $(LDFLAGS) $(SC_LIBS) -o "$@"

clean:
	rm -f "$(TARGET)"
