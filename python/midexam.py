# -*- coding: utf-8 -*-

def sortThreeNum(a,b,c):
    if (a < b):
        t = a;
        a = b;
        b = t;
    elif (b < c):
        t = b;
        b = c;
        c = t;
    elif (a < c) :
        t = a;
        a = c;
        c = t;
    return a,b,c;

print(sortThreeNum(3,5,9))