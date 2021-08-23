# The Art of C++
# Copyright (c) 2015-2019 Daniel Frey
# Please see LICENSE for license or visit https://github.com/taocpp/sequences/

.SUFFIXES:
.SECONDARY:

ifeq ($(OS),Windows_NT)
UNAME_S := $(OS)
else
UNAME_S := $(shell uname -s)
endif

# For Darwin (Mac OS X / macOS) we assume that the default compiler
# clang++ is used; when $(CXX) is some version of g++, then
# $(CXXSTD) has to be set to -std=c++11 (or newer) so
# that -stdlib=libc++ is not automatically added.

ifeq ($(CXXSTD),)
CXXSTD := -std=c++11
ifeq ($(UNAME_S),Darwin)
CXXSTD += -stdlib=libc++
endif
endif

# Ensure strict standard compliance and no warnings, can be
# changed if desired.

CPPFLAGS ?= -pedantic
CXXFLAGS ?= -Wall -Wextra -Wshadow -Werror -O3

CLANG_TIDY ?= clang-tidy

HEADERS := $(shell find include -name '*.hpp')
SOURCES := $(shell find src -name '*.cpp')
DEPENDS := $(SOURCES:%.cpp=build/%.d)
BINARIES := $(SOURCES:%.cpp=build/%)

UNIT_TESTS := $(filter build/src/test/%,$(BINARIES))

.PHONY: all
all: compile check

.PHONY: compile
compile: $(BINARIES)

.PHONY: check
check: $(UNIT_TESTS)
	@set -e; for T in $(UNIT_TESTS); do echo $$T; $$T > /dev/null; done
	@build/src/test/seq/tester

.PHONY: clean
clean:
	@rm -rf build
	@find . -name '*~' -delete

build/%.d: %.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXSTD) -Iinclude $(CPPFLAGS) -MM -MQ $@ $< -o $@

build/%: %.cpp build/%.d
	$(CXX) $(CXXSTD) -Iinclude $(CPPFLAGS) $(CXXFLAGS) $< -o $@

build/%.clang-tidy: % .clang-tidy
	$(CLANG_TIDY) -quiet $< -- $(CXXSTD) -Iinclude $(CPPFLAGS) $(CXXFLAGS) 2>/dev/null
	@mkdir -p $(@D)
	@touch $@

.PHONY: clang-tidy
clang-tidy: $(HEADERS:%=build/%.clang-tidy) $(SOURCES:%=build/%.clang-tidy)
	@echo "All $(words $(HEADERS) $(SOURCES)) clang-tidy tests passed."

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(DEPENDS)
endif
