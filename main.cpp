#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include "third_party/libsvm/svm.h"

// Compute RMSE on training data
static double rmse(const std::vector<double>& y,
                   const std::vector<double>& yhat)
{
    if (y.size() != yhat.size() || y.empty()) return NAN;
    double s = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        double e = yhat[i] - y[i];
        s += e * e;
    }
    return std::sqrt(s / static_cast<double>(y.size()));
}

int main()
{
    // Tabular data: Mach -> Kmax
    std::vector<double> M    = {0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5};
    std::vector<double> Kmax = {14.2, 13.8, 13.1, 12.0, 10.5,  9.1,  8.0};

    const int N = static_cast<int>(M.size());
    if (N != static_cast<int>(Kmax.size()) || N < 2) {
        std::cerr << "Invalid dataset size\n";
        return 1;
    }

    // Prepare libsvm problem
    svm_problem prob;
    prob.l = N;
    prob.y = new double[N];
    prob.x = new svm_node*[N];

    for (int i = 0; i < N; ++i) {
        prob.y[i] = Kmax[i];

        prob.x[i] = new svm_node[2];
        prob.x[i][0].index = 1;
        prob.x[i][0].value = M[i];
        prob.x[i][1].index = -1;
        prob.x[i][1].value = 0.0;
    }

    // SVR parameters
    svm_parameter param;
    std::memset(&param, 0, sizeof(param));

    param.svm_type = EPSILON_SVR;
    param.kernel_type = RBF;

    param.gamma = 0.5;
    param.C = 100.0;
    param.p = 0.05;

    param.cache_size = 100;
    param.eps = 1e-3;
    param.shrinking = 1;
    param.probability = 0;

    // Required by libsvm
    param.degree = 3;
    param.coef0 = 0.0;
    param.nu = 0.5;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;

    const char* err = svm_check_parameter(&prob, &param);
    if (err) {
        std::cerr << "SVR parameter error: " << err << "\n";
        return 1;
    }

    // Train model
    svm_model* model = svm_train(&prob, &param);

    // Training error
    std::vector<double> yhat;
    yhat.reserve(N);

    svm_node x[2];
    x[0].index = 1;
    x[1].index = -1;
    x[1].value = 0.0;

    for (int i = 0; i < N; ++i) {
        x[0].value = M[i];
        yhat.push_back(svm_predict(model, x));
    }

    std::cout << "Train RMSE: " << rmse(Kmax, yhat) << std::endl;

    // Predict on dense grid and save
    std::ofstream out("predictions.tsv");
    out << "Mach\tKmax_pred\n";

    for (double m = 0.3; m <= 1.5 + 1e-12; m += 0.05) {
        x[0].value = m;
        double k = svm_predict(model, x);
        out << m << "\t" << k << "\n";
    }
    out.close();

    // Save trained model
    svm_save_model("Kmax_SVR.model", model);

    // Cleanup
    svm_free_and_destroy_model(&model);
    svm_destroy_param(&param);

    for (int i = 0; i < N; ++i) delete[] prob.x[i];
    delete[] prob.x;
    delete[] prob.y;

    return 0;
}
