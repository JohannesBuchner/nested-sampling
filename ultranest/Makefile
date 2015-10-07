
CC := gcc
CFLAGS +=  -std=c99 -Wall -Wextra -Wno-unused-parameter -pedantic -O3

all: libultranest.so example.exe

doc:
	mkdir doc/api -p
	doxygen Doxyfile && make -C doc/api/latex/
	@echo
	@echo "HTML documentation at doc/api/html/index.html"
	@echo "PDF  documentation at doc/api/latex/refman.pdf"

libultranest.so: ultranest.c sampler.c draw.c common.c progressbar/libprogressbar.so
	${CC} ${CFLAGS} -I progressbar/include/ -lncurses -fPIC -shared -o $@ $^

example.exe: example.c libultranest.so
	${CC} ${CFLAGS} -lm -lgsl -o $@ $^

.PHONY: all doc

