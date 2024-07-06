#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    float **wxh;
    float **whh;
    float **who;
    float *bh;
    float *bo;
} RNN;

void initializeParams(RNN *rnn, int inputSize, int hiddenLayers, int outputSize){
    rnn->wxh = (float **)malloc(hiddenLayers * sizeof(float *));
    for (int i = 0; i < hiddenLayers; i++) {
        rnn->wxh[i] = (float *)malloc(inputSize * sizeof(float));
        for(int j = 0; j < inputSize; j++){
            rnn->wxh[i][j] = rand() * 0.0001;
        }
    }

    rnn->whh = (float **)malloc(hiddenLayers * sizeof(float *));
    for (int i = 0; i < hiddenLayers; i++) {
        rnn->whh[i] = (float *)malloc(hiddenLayers * sizeof(float));
        for(int j = 0; j < hiddenLayers; j++){
            rnn->whh[i][j] = rand() * 0.0001;
        }
    }

    rnn->who = (float **)malloc(outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++) {
        rnn->who[i] = (float *)malloc(hiddenLayers * sizeof(float));
        for(int j = 0; j < hiddenLayers; j++){
            rnn->who[i][j] = rand() * 0.0001;
        }
    }

    rnn->bh = (float *)malloc(hiddenLayers * sizeof(float));
    for (int i = 0; i < hiddenLayers; i++) {
        rnn->bh[i] = rand() * 0.0001;
    }

    rnn->bo = (float *)malloc(sizeof(float));
    *rnn->bo = rand() * 0.0001;
}

float *dot(float **w, float *x, int hiddenLayers, int inputSize){
    float *matrix = (float *)malloc(hiddenLayers * sizeof(float));
    for(int i = 0; i < hiddenLayers; i ++){
        matrix[i] = 0.0;
        for(int j = 0; j < inputSize; j++){
            matrix[i] += w[i][j] * x[j];
        }
    }
    return matrix;
}

int main() {
    float input[] = {1.0, 0.5, -1.0};
    int inputSize = sizeof(input) / sizeof(float);
    int hiddenLayers;
    int outputSize = 1;
    printf("Enter Hidden Layers: ");
    scanf("%d", &hiddenLayers);
    srand(time(NULL));
    RNN rnn;
    initializeParams(&rnn, inputSize, hiddenLayers, outputSize);
    float *z1 = dot(rnn.wxh, input, hiddenLayers, inputSize);
    for(int i = 0; i < hiddenLayers; i++){
        for(int j = 0; j < inputSize; j++){
            printf("%f\n", rnn.wxh[i][j]);
        }
    }
    for(int i = 0; i < hiddenLayers; i++){
        printf("%lu\n", z1[i]);
    }

    // int **a;
    // a = (int **)malloc(3 * sizeof(int *));
    // for(int i = 0; i < 3; i++){
    //     a[i] = (int *)malloc(3 * sizeof(int));
    //     for(int j = 0; j < 3; j++){
    //         a[i][j] = 3;
    //     } 
    // }
    
    // int ***b = &a;
    // for(int i = 0; i < 3; i++){
    //     for(int j = 0; j < 3; j++){
    //         printf("%d\n", b[i][j]);
    //     }
    // }
}