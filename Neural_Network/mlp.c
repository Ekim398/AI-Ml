#include "mlp.h"
#include <stdlib.h>
#include <math.h>

double random_weight(double a, double b) {
    return a + (b - a) * ((double)rand() / (double)RAND_MAX);
}

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double dsigmoid_from_a(double a) {
    return a * (1.0 - a);
}

void mlp_init(MLP *net) {//Input -> H1: W*X(input)[2][1]
    for(int m=0; m < h1_dim; m++) { //2 input to 3 neurons in h_1
        net-> bias_1[m]=0.0;

        for(int n=0; n <input_dim; n++) {
            net->weight_1[m][n] = random_weight(-1.0, 1.0) * 0.5;
        }
    }

    //H1 -> H2: w2 * h1  
    for(int i=0; i < h2_dim; i++) {//3 neurons -> 2 neurons
        net -> bias_2[i]=0.0;
    
        for(int j=0; j < h1_dim; j++) {
            net -> weight_2[i][j] = random_weight(-1.0, 1.0) * 0.5;
        }
    }

    //H2 -> Output 
    for(int p=0; p< output_dim; p++){
        net -> bias_3[p]=0.0;

        for(int q=0; q<h2_dim; q++) {
            net -> weight_3[p][q] = random_weight(-1.0, 1.0) * 0.5;
        }
    }
}

void forward_pass(
    const MLP *net, //structure should be the same
    const double x[input_dim],
    double z1[h1_dim],
    double a1[h1_dim],
    double z2[h2_dim],
    double a2[h2_dim],
    double z3[output_dim],
    double output[output_dim]
) {
    //z1 = W1 *x + b
    for(int m=0; m < h1_dim; m++) { //2 input to 3 neurons in h_1
        double sum = net -> bias_1[m]; //3 bias

        for(int n=0; n <input_dim; n++) {
            sum += net -> weight_1[m][n] * x[n]; //weight(3X2) * input(2X1) = 3X1
        }
        //z1, a 
        z1[m] = sum; //3X1
        a1[m] = sigmoid(z1[m]); //3X1
    }

    //z2= W2 * h1  
    for(int i=0; i < h2_dim; i++) {//3 neurons -> 2 neurons
        double sum = net -> bias_2[i];
    
        for(int j=0; j < h1_dim; j++) {
            sum += net -> weight_2[i][j] * a1[j];
        }
        z2[i] = sum;
        a2[i] = sigmoid(z2[i]);
    }

    //z3 = W3 * h2 
    for(int p=0; p< output_dim; p++){
        double sum = net -> bias_3[p];

        for(int q=0; q<h2_dim; q++) {
            sum += net -> weight_3[p][q] * a2[q];
        }
        z3[p] = sum;
        output[p] = sigmoid(z3[p]);
    }
} 

double mlp_train(
    MLP *net,
    const double x[input_dim],
    const double y[output_dim],
    double lr
) {
    //A  -> Forward pass
    double z1[h1_dim], a1[h1_dim];
    double z2[h2_dim], a2[h2_dim];
    double z3[output_dim], output[output_dim];
    forward_pass(net, x, z1, a1, z2, a2, z3, output);

    //loss MSE
    double loss = 0.0;
    for (int p = 0; p < output_dim; p++) {
        double difference = output[p] - y[p];
        loss += 0.5 * difference * difference; 
    }

    //B  -> Backpropagation (Backwards pass)

    //1.A Output Layer: z3 -> W3 * h2
    //dL/dz2 = (output - y) * dsigmoid(z2)
    double dZ3[output_dim]; //Grad W2, Z2
    for (int p=0; p < output_dim; p++) {
        double diff = output[p] - y[p];
        dZ3[p] = diff * dsigmoid_from_a(output[p]); 
    }

    //1.B Outputer layer -> hidden layer 2: Gradients dL/dW3,b3
    double dW3[output_dim][h2_dim];
    double db3[output_dim];
    for (int p = 0; p < output_dim; p++) {
        db3[p] = dZ3[p];
        for (int q = 0; q < h2_dim; q++) {
            dW3[p][q] = dZ3[p] * a2[q];
        }
    }
    
    //2.A Hidden layer 2: dL/da2, dL/dz2 
    double dA2[h2_dim];
    for (int q = 0; q < h2_dim; q++) {
        double sum = 0.0;
        for (int p = 0; p < output_dim; p++) {
            sum += net->weight_3[p][q] * dZ3[p];
        }
        dA2[q] = sum;
    }

    double dZ2[h2_dim];
    for (int i = 0; i < h2_dim; i++) {
        dZ2[i] = dA2[i] * dsigmoid_from_a(a2[i]);
    }

    //2.B Hidden layer 2 -> Hidden layer 1: Gradients dL/dW2, b2 
    double dW2[h2_dim][h1_dim];
    double db2[h2_dim];
    for (int i = 0; i < h2_dim; i++) {
        db2[i] = dZ2[i];
        for (int j = 0; j < h1_dim; j++) {
            dW2[i][j] = dZ2[i] * a1[j];
        }
    }

    //3.A Hidden layer 1: DL/da1, dL/dz1
    double dA1[h1_dim];
    for (int j = 0; j < h1_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < h2_dim; i++) {
            sum += net->weight_2[i][j] * dZ2[i];
        }
        dA1[j] = sum;
    }

    double dZ1[h1_dim];
    for (int m = 0; m < h1_dim; m++) {
        dZ1[m] = dA1[m] * dsigmoid_from_a(a1[m]);
    }

    //3.B Hidden Layer 1 -> Input Layer: dL/dW1, b1
    double dW1[h1_dim][input_dim];
    double db1[h1_dim];
    for (int m = 0; m < h1_dim; m++) {
        db1[m] = dZ1[m];
        for (int n = 0; n < input_dim; n++) {
            dW1[m][n] = dZ1[m] * x[n];
        }
    }

    //Stochastic gradient descent SGD for optimizer
    for (int m = 0; m < h1_dim; m++) {
        net->bias_1[m] -= lr * db1[m];
        for (int n = 0; n < input_dim; n++) {
            net->weight_1[m][n] -= lr * dW1[m][n];
        }
    }

    for (int i = 0; i < h2_dim; i++) {
        net->bias_2[i] -= lr * db2[i];
        for (int j = 0; j < h1_dim; j++) {
            net->weight_2[i][j] -= lr * dW2[i][j];
        }
    }

    for (int p = 0; p < output_dim; p++) {
        net->bias_3[p] -= lr * db3[p];
        for (int q = 0; q < h2_dim; q++) {
            net->weight_3[p][q] -= lr * dW3[p][q];
        }
    }

    return loss;
}

