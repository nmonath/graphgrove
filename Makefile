# Modified from Makefile of CoverTree
# https://github.com/manzilzaheer/CoverTree
#
# Copyright (c) 2017 Manzil Zaheer All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CURR_DIR = $(shell pwd)
SOURCES = $(wildcard $(CURR_DIR)/src/*)
BUILDDIR = $(subst $(CURR_DIR)/src/,$(CURR_DIR)/build/,$(SOURCES))
PROGS = $(subst $(CURR_DIR)/src/,,$(SOURCES))
CLEAN_PROGS = $(subst $(CURR_DIR)/src/,clean-,$(SOURCES))


CTYPE = gcc

.PHONY: all dir compile $(SOURCES)

all: dir compile py

py:
	@python setup.py install

intel: CTYPE = intel
intel: dir compile

inteltogether: CTYPE = inteltogether
inteltogether: dir compile

llvm: CTYPE = llvm
llvm: dir compile


dir:
	@echo Setting up directories
	@mkdir -p $(BUILDDIR)
	@mkdir -p dist


compile: $(SOURCES)

$(SOURCES): $(CURR_DIR)/src/% : $(CURR_DIR)/src/%/makefile
	@cd $@ && make $(CTYPE) SOURCEDIR=$@ BUILDDIR=$(CURR_DIR)/build/$* EXECUTABLE=$(CURR_DIR)/dist/$*
	@echo

clean:
	rm -rf $(BUILDDIR)
	rm -rf dist/*

$(PROGS): % : $(CURR_DIR)/src/%/makefile
	@mkdir -p $(CURR_DIR)/build/$@
	@cd $(CURR_DIR)/src/$@ && make $(CTYPE) SOURCEDIR=$(CURR_DIR)/src/$@ BUILDDIR=$(CURR_DIR)/build/$@ EXECUTABLE=$(CURR_DIR)/dist/$@
	@echo

$(CLEAN_PROGS): clean-% : $(CURR_DIR)/src/%/makefile
	rm -rf build/$(subst clean-,,$@)
	rm -rf dist/$(subst clean-,,$@)


docs-build:
	cp README.md homepage/_includes/
	cd homepage && bundle exec jekyll build -d ../docs/build/html
	sphinx-build -b html docs/source/ docs/build/html/documentation

docs-serve: docs-build
	cd homepage && bundle exec jekyll serve -d ../docs/build/html --skip-initial-build

docs-deploy: docs-build
	cd docs && make gh-deploy
