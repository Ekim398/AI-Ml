#ifndef MLP_H
#define MLP_H

#include <stddef.h>

#define input_dim  2
#define h1_dim     3
#define h2_dim     2
#define output_dim 1

typedef struct {
    double weight_1[h1_dim][input_dim]; //Weight1[3][2] X Input[2][1] = Hidden1[3][1] + bias[3][1]
    double bias_1[h1_dim]; //each column of weights[m][n] corresponds to each input variable in input[m] 
    //n_weight = # of columns of weights = m_input = # of rows of inputs(x1, x2) 
    //m_weight = # of rows in weights = # of neurons in hidden layer = # of bias rows
    double weight_2[h2_dim][h1_dim]; //Weight2[2][3] X Hidden1[3][1] = Hidden2[2][1] + bias[2][1]
    double bias_2[h2_dim];

    double weight_3[output_dim][h2_dim]; //weight3[1][2] X Hidden2[2][1] = output[1][1] + bias[1][1]
    double bias_3[output_dim];

} MLP;

void mlp_init(MLP *net);

void forward_pass(
    const MLP *net,
    const double x[input_dim],
    double z1[h1_dim],
    double a1[h1_dim],
    double z2[h2_dim],
    double a2[h2_dim],
    double z3[output_dim],
    double output[output_dim]
);

double mlp_train(
    MLP *net,
    const double x[input_dim],
    const double y[output_dim],
    double lr
);

double sigmoid(double z);
double dsigmoid_from_a(double a);
double random_weight(double a, double b);

#endif
