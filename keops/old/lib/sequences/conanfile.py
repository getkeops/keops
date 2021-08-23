#!/usr/bin/env python
# -*- coding: utf-8 -*-
from conans import ConanFile, CMake

class SequencesConan(ConanFile):
    name = "sequences"
    description = "C++11 header-only library that provides efficient algorithms to generate and work on variadic templates and std::integer_sequence"
    homepage = "https://github.com/taocpp/sequences"
    url = homepage
    license = "MIT"
    author = "taocpp@icemx.net"
    exports = "LICENSE"
    exports_sources = "include/*", "CMakeLists.txt"
    no_copy_source = True

    def build(self):
        pass

    def package(self):
        cmake = CMake(self)

        cmake.definitions["TAOCPP_SEQUENCES_BUILD_TESTS"] = "OFF"
        cmake.definitions["TAOCPP_SEQUENCES_INSTALL_DOC_DIR"] = "licenses"

        cmake.configure()
        cmake.install()

    def package_id(self):
        self.info.header_only()
