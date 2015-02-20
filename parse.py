#!/usr/bin/env python

from bs4 import BeautifulSoup
from os import listdir
from collections import Counter, defaultdict
import sys
import pickle

source_dir = sys.argv[1]

tags = Counter()
scp_tags = defaultdict(list)
links = defaultdict(set)
for file_name in listdir(source_dir):
    if not file_name[0:3] == 'scp': continue
    num = file_name.split('-')[1]

    soup = BeautifulSoup(open('%s/%s' % (source_dir, file_name)))

    # Is the page a placeholder?
    if not 'SCP' in soup.find(id = 'page-title').string: continue
    
    # Extract links
    content = soup.find('div', id = 'page-content')
    for content_anchor in content.find_all('a'):
        if not 'href' in content_anchor.attrs: continue
        href = content_anchor['href']
        if href == 'javascript:;': continue
        if href[0:5] == '/scp-':
            target = href[5:]
            target = target.split('-')[0]
            if target == num: continue
            links[num].add(target)
    
    tag_content = soup.find('div', {'class': 'page-tags'})
    for tag_anchor in tag_content.find_all('a'):
        tag = tag_anchor.string[:]
        if tag == 'scp': continue
        tags[tag] += 1
        scp_tags[num].append(tag)
        
print tags
print scp_tags
print links

# Pickle parsed tags and network for later use
output = open('%s/parse.pkl' % source_dir, 'wb')
pickle.dump({'tags': tags, 'scp_tags': scp_tags, 'links': links}, output)
output.close()
