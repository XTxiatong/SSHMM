from __future__ import print_function
import sys
from itertools import groupby


def read_from_text(file):
    for line in file:
        yield line.strip('\r\n').split('\t')


def main():
    for key, values in groupby(read_from_text(sys.stdin), key=lambda x: x[0]):
        value_sum = sum((int(v[1]) for v in values))
        print('%s\t%d' % (key, value_sum))


if __name__ == "__main__":
    main()
