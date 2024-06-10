// y = mx + b
// m = nΣxy - ΣxΣy / nΣx^2 - (Σx)^2
// b = Σy - mΣx / n

#include <iostream>
using namespace std;

void linearRegression(float x[], float y[], int n){
    float sumx = 0;
    float sumy = 0;
    float sumxy = 0;
    float sumxsq = 0;

    for(int i = 0; i < n; i++){
        sumx += x[i];
        sumy += y[i];
        sumxy += x[i] * y[i];
        sumxsq += x[i] * x[i];
    }

    cout << "sum of x = " << sumx << endl << "sum of y = " << sumy << endl << "sum of x * y = " << sumxy << endl << "sum of x square= " << sumxsq << endl;

    float m = ((n * sumxy) - (sumx * sumy)) / ((n * sumxsq) - (sumx * sumx));
    cout << "m = " << m << endl;

    float b = (sumy - (m * sumx)) / n;
    cout << "b = " << b << endl;
    cout << "The equation becomes y = " << m << "x + " << b;  
}

int main() {
    float x[7] = {1, 2, 3, 4, 5, 6, 7};
    float y[7] = {1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16};
    linearRegression(x, y, 7);
    return 0;
}