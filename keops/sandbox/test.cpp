// test convolution
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++11 -O2 -o build/test test.cpp
// 

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iostream>

#include "core/Pack.h"


using namespace std;


int main() {
    using p = pack<3,4,0,2,0>;
    cout << "pack : ";
    p::PrintId();
    cout << endl;

    cout << "TAB : ";
    CheckAllDistinct<p>::TAB::PrintId();
    cout << endl;

    cout << "CheckAllDistinct : " << CheckAllDistinct<p>::val << endl;
}



