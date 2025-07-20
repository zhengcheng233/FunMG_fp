#!/usr/bin/env python
import os

dirlist = []
for root,dirs,files in os.walk('./'):
    for d in dirs:
        if d.endswith("evc"):
            dirlist.append(os.path.join(root,d))
    for f in files:
        if f.endswith("chk"):
            dirlist.append(os.path.join(root,f))

#print(dirlist)
for dir0 in dirlist:
    os.system('rm -rf ' +dir0)
