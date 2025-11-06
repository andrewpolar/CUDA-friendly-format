#pragma once
#include <memory>
#include <vector>
#include <random>
#include <ctime>

struct UrysohnModel {
	std::vector<std::vector<double>> model;
	std::vector<double> xmin;
	std::vector<double> xmax;
	std::vector<double> deltax;
	std::vector<double> offset;
	std::vector<int>    index;
};

void InitializeUrysohnModel(UrysohnModel& u, int nFunctions, int nPoints, double xmin, double xmax, double fmin, double fmax) {
	u.model.resize(nFunctions, std::vector<double>(nPoints));
	u.xmin.resize(nFunctions);
	u.xmax.resize(nFunctions);
	u.deltax.resize(nFunctions);
	u.offset.resize(nFunctions);
	u.index.resize(nFunctions);

	std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
	std::uniform_real_distribution<double> dist(fmin, fmax);
	for (int i = 0; i < nFunctions; ++i) {
		for (int j = 0; j < nPoints; ++j) {
			u.model[i][j] = dist(rng);
		}
	}

	for (int i = 0; i < nFunctions; ++i) {
		u.xmin[i] = xmin;
		u.xmax[i] = xmax;
		double gap = 0.01 * (u.xmax[i] - u.xmin[i]);
		u.xmin[i] -= gap;
		u.xmax[i] += gap;
		u.deltax[i] = (u.xmax[i] - u.xmin[i]) / (nPoints - 1);
	}
};

class UrysohnLogic {
public:
	UrysohnLogic(UrysohnModel& reference): U(&reference) {
	}
	double Compute(const std::vector<double>& inputs, std::vector<double>& derivatives, bool freezeModel) {
		double f = 0.0;
		for (int i = 0; i < (int)U->model.size(); ++i) {
			f += Compute(i, inputs[i], derivatives[i], freezeModel);
		}
		return f / (double)U->model.size();
	}
	double Compute(const std::vector<double>& inputs, bool freezeModel) {
		double f = 0.0;
		for (size_t i = 0; i < U->model.size(); ++i) {
			f += Compute(i, inputs[i], freezeModel);
		}
		return f / (double)U->model.size();
	}
	void Update(double delta) {
		for (int i = 0; i < (int)U->model.size(); ++i) {
			Update(i, delta);
		}
	}

private:
	UrysohnModel* U;  

	void SetLimits(size_t k) {
		double gap = 0.01 * (U->xmax[k] - U->xmin[k]);
		U->xmin[k] -= gap;
		U->xmax[k] += gap;
		U->deltax[k] = (U->xmax[k] - U->xmin[k]) / (U->model[k].size() - 1);
	}
	void Update(int k, double residual) {
		double tmp = residual * U->offset[k];
		U->model[k][U->index[k] + 1] += tmp;
		U->model[k][U->index[k]] += residual - tmp;
	}
	double Compute(int k, double x, double& derivative, bool freezeModel) {
		double f = Compute(k, x, freezeModel);
		derivative = GetDerivative(k);
		return f;
	}
	double GetDerivative(int k) {
		return (U->model[k][U->index[k] + 1] - U->model[k][U->index[k]]) / U->deltax[k];
	}
	double Compute(size_t k, double x, bool freezeModel) {
		if (!freezeModel) {
			FixLimits(k, x);
		}
		else {
			if (x <= U->xmin[k]) {
				U->index[k] = 0;
				U->offset[k] = 0.001;
				return U->model[k][0];
			}
			if (x >= U->xmax[k]) {
				U->index[k] = (int)U->model[k].size() - 2;
				U->offset[k] = 0.999;
				return U->model[k][U->model[k].size() - 1];
			}
		}
		double R = (x - U->xmin[k]) / U->deltax[k];
		U->index[k] = (int)(R);
		U->offset[k] = R - U->index[k];
		return U->model[k][U->index[k]] + (U->model[k][U->index[k] + 1] - U->model[k][U->index[k]]) * U->offset[k];
	}
	void FixLimits(size_t k, double x) {
		if (x <= U->xmin[k]) {
			U->xmin[k] = x;
			SetLimits(k);
		}
		if (x >= U->xmax[k]) {
			U->xmax[k] = x;
			SetLimits(k);
		}
	}
};

