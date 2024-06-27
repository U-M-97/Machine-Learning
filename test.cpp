#include <iostream>

int main() {
    const int size = 10; // Assuming size is defined as 10
    double array1[size] = {0.10318736,  0.09828043,  0.09699903,  0.11148215,  0.10214534, -0.90603846,
   0.107077,    0.11208093 , 0.10192947 , 0.07285676};

    double sum = 0.0;

    // Calculate the sum of all elements in array1
    for (int i = 0; i < size; ++i) {
        sum += array1[i];
    }

    // Output the sum
    std::cout << "Sum of array elements: " << sum << std::endl;

    return 0;
}
