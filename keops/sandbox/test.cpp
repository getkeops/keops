// test convolution 
// compile with 
//    g++ -I.. -D__TYPE__=float -std=c++11 -O2 -o build/test test.cpp 
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
    float a[20];
    for (int k=0; k<20; k++)
    {
        a[k] = k+.1;
    }
    cout << Get<14>(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],a[19]) << endl;
} 
 
