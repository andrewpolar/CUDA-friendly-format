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

void InitializeUrysohnModel(UrysohnModel& u, int nFunctions, int nPoints, 
	double xmin, double xmax, double fmin, double fmax, std::mt19937& rng) {
	u.model.resize(nFunctions, std::vector<double>(nPoints));
	u.xmin.resize(nFunctions);
	u.xmax.resize(nFunctions);
	u.deltax.resize(nFunctions);
	u.offset.resize(nFunctions);
	u.index.resize(nFunctions);

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

double Compute(const std::vector<double>& inputs, bool freezeModel, UrysohnModel& U) {
	double f = 0.0;
	for (size_t i = 0; i < U.model.size(); ++i) {
		double x = inputs[i];
		if (!freezeModel) {
			bool isChanged = false;
			if (x <= U.xmin[i]) {
				U.xmin[i] = x;
				isChanged = true;
			}
			else if (x >= U.xmax[i]) {
				U.xmax[i] = x;
				isChanged = true;
			}
			if (isChanged) {
				double gap = 0.01 * (U.xmax[i] - U.xmin[i]);
				U.xmin[i] -= gap;
				U.xmax[i] += gap;
				U.deltax[i] = (U.xmax[i] - U.xmin[i]) / (U.model[i].size() - 1);
			}
		}
		if (x <= U.xmin[i]) {
			U.index[i] = 0;
			U.offset[i] = 0.001;
			f += U.model[i][0];
		}
		else if (x >= U.xmax[i]) {
			U.index[i] = (int)U.model[i].size() - 2;
			U.offset[i] = 0.999;
			f += U.model[i][U.model[i].size() - 1];
		}
		else {
			double R = (x - U.xmin[i]) / U.deltax[i];
			U.index[i] = (int)(R);
			U.offset[i] = R - U.index[i];
			f += U.model[i][U.index[i]] + (U.model[i][U.index[i] + 1] - U.model[i][U.index[i]]) * U.offset[i];
		}
	}
	return f / (double)U.model.size();
}

void ComputeDerivatives(std::vector<double>& derivatives, UrysohnModel& U) {
	for (int i = 0; i < (int)U.model.size(); ++i) {
		derivatives[i] = (U.model[i][U.index[i] + 1] - U.model[i][U.index[i]]) / U.deltax[i];
	}
}

void Update(double delta, UrysohnModel& U) {
	for (int i = 0; i < (int)U.model.size(); ++i) {
		double tmp = delta * U.offset[i];
		U.model[i][U.index[i] + 1] += tmp;
		U.model[i][U.index[i]] += delta - tmp;
	}
}
