#!/usr/bin/env python

params = { 'seconds_between': 20 }

import sys
from os import system
from time import sleep

m, n, dest_dir = sys.argv[1:4]
m, n = int(m), int(n)

for i in range(m,n+1):
    if 0 < i < 1000:
        num = '%03d' % i
    else:
        num = '%04d' % i

    url = 'scp-wiki.wikidot.com/scp-' + num

    system('curl %s > %s/scp-%s' % (url, dest_dir, num))
    sleep(params['seconds_between'])
