//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

//This is sequential Newton-Kaczmarz method for Kolmogorov-Arnold networks. The features are random matrices,
//targets are their determinants. Accuracy metric is Pearson correlation coefficient.
//This code is restructured for CUDA friendly format. The model is set of structures and 
//logic is isolated from model.

//Although this code is sequential, typical Windows execution is faster than FastKAN
//Pearson 0.839, Time 0.734
//Pearson 0.910, Time 1.472
//Pearson 0.935, Time 2.213
//Pearson 0.945, Time 2.952
//Pearson 0.955, Time 3.697
//Pearson 0.962, Time 4.433
//Pearson 0.964, Time 5.178
//Pearson 0.967, Time 5.918
//Pearson 0.969, Time 6.655
//Pearson 0.971, Time 7.406

//FastKAN needs 41 seconds for CPU and 11 seconds for GPU on the same machine. 

#include <iostream>
#include <cmath>
#include "Helper.h"
#include "Urysohn.h"

//configuration
const int nInner = 64;
const double alpha = 0.1;

// global buffers (keep same as before)
extern std::vector<double> models(nInner);
extern std::vector<double> deltas(nInner);

// ComputeInner — runs each inner logic model
void ComputeInner(const std::vector<std::unique_ptr<UrysohnLogic>>& uInnerLogic, const std::vector<double>& features) {
	for (int i = 0; i < nInner; ++i) {
		models[i] = uInnerLogic[i]->Compute(features, false);
	}
}

// DoOuter — runs outer logic and prepares deltas
void DoOuter(const std::unique_ptr<UrysohnLogic>& uOuterLogic, double target) {
	static std::vector<double> derivatives(nInner);
	double prediction = uOuterLogic->Compute(models, derivatives, false);
	double residual = alpha * (target - prediction);
	for (int i = 0; i < nInner; ++i) {
		deltas[i] = derivatives[i] * residual;
	}
	uOuterLogic->Update(residual);
}

// UpdateInner — applies deltas back to inner logic models
void UpdateInner(std::vector<std::unique_ptr<UrysohnLogic>>& uInnerLogic) {
	for (size_t i = 0; i < uInnerLogic.size(); ++i) {
		uInnerLogic[i]->Update(deltas[i]);
	}
}

int main() {
	int nTrainingRecords = 100'000;
	int nValidationRecords = 20'000;
	int nMatrixSize = 4;
	int nFeatures = nMatrixSize * nMatrixSize;
	double min = 0.0;
	double max = 10.0;
	auto features_training = Helper::GenerateInput(nTrainingRecords, nFeatures, min, max);
	auto features_validation = Helper::GenerateInput(nValidationRecords, nFeatures, min, max);
	auto targets_training = Helper::ComputeDeterminantTarget(features_training, nMatrixSize);
	auto targets_validation = Helper::ComputeDeterminantTarget(features_validation, nMatrixSize);

	clock_t start_application = clock();
	clock_t current_time = clock();

	//find limits
	std::vector<double> argmin;
	std::vector<double> argmax;
	for (int i = 0; i < nFeatures; ++i) {
		argmin.push_back(min);
		argmax.push_back(max);
	}

	double targetMin = Helper::MinV(targets_training);
	double targetMax = Helper::MaxV(targets_training);

	// Instantiate models
	std::vector<std::unique_ptr<UrysohnModel>> uInnerModels;
	std::vector<std::unique_ptr<UrysohnLogic>> uInnerLogic;
	for (int i = 0; i < nInner; ++i) {
		auto model = std::make_unique<UrysohnModel>();
		InitializeUrysohnModel(*model, nFeatures, 3, min, max, targetMin, targetMax);
		uInnerLogic.push_back(std::make_unique<UrysohnLogic>(*model));
		uInnerModels.push_back(std::move(model));
	}

	// Outer model
	auto uOuterModel = std::make_unique<UrysohnModel>();
	InitializeUrysohnModel(*uOuterModel, nInner, 30, targetMin, targetMax, targetMin, targetMax);
	auto uOuterLogic = std::make_unique<UrysohnLogic>(*uOuterModel);

	//training
	for (int epoch = 0; epoch < 16; ++epoch) {
		for (int record = 0; record < nTrainingRecords; ++record) {
			ComputeInner(uInnerLogic, features_training[record]);
			DoOuter(uOuterLogic, targets_training[record]);
			UpdateInner(uInnerLogic);
		}

		//validation
		auto v = std::vector<double>(nValidationRecords);
		for (int record = 0; record < nValidationRecords; ++record) {
			//compute
			for (size_t i = 0; i < uInnerLogic.size(); ++i) {
				models[i] = uInnerLogic[i]->Compute(features_validation[record], true);
			}
			v[record] = uOuterLogic->Compute(models, true);
		}

		double pearson = Helper::Pearson(v, targets_validation);

		current_time = clock();
		printf("Pearson %4.3f, Time %2.3f\n", pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);
		if (pearson >= 0.97) break;
	}
}

