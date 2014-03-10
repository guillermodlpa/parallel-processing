
#include <stdio.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) 
{	

	int N=5;

	float A [5][5];

	for ( int i = 0; i < N; i++ ) 
		for ( int j = 0; j < N; j++ )
			A[i][j] = i+j;

	cout << "start" << endl;

	cout << "A[1][1]=" << A[1][1] << endl;

	cout << "A[1][1]=" << *((*A)+1+N) << endl;
}