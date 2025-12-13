#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mlp.h"

static double predict_one(MLP *net, const double x[input_dim]) {
    double z1[h1_dim], a1[h1_dim];
    double z2[h2_dim], a2[h2_dim];
    double z3[output_dim], out[output_dim];

    forward_pass(net, x, z1, a1, z2, a2, z3, out);
    return out[0];
}

int main(void) {
    printf("Hello we are going to be coding a basic MLP neural network from scratch in C\n");

    srand((unsigned int)time(NULL));

    MLP net;
    mlp_init(&net);
    
    printf("W3[0][0] before = %.6f\n", net.weight_3[0][0]);

    // XOR dataset
    const double X[4][input_dim] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    const double Y[4][output_dim] = {
        {0},
        {1},
        {1},
        {0}
    };

    double lr = 0.5;
    int epochs = 30000;

    for (int e = 1; e <= epochs; e++) {
        double total_loss = 0.0;
        for (int n = 0; n < 4; n++) {
            total_loss += mlp_train(&net, X[n], Y[n], lr);
        }
        if (e % 30000 == 0) {
            printf("epoch %d | avg loss = %.6f\n", e, total_loss / 4.0);
        }
    }

    printf("\nPredictions:\n");
    for (int n = 0; n < 4; n++) {
        double p = predict_one(&net, X[n]);
        printf("x=(%.0f,%.0f) -> yhat=%.4f (target=%.0f)\n",
               X[n][0], X[n][1], p, Y[n][0]);
    }

    return 0;
}
