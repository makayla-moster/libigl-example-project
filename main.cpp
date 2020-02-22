#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/opengl/glfw/Viewer.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <iostream>
//#include "tutorial_shared_path.h"

Eigen::MatrixXd V, U, CV;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L, K;
igl::opengl::glfw::Viewer viewer;

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

	srand(time(NULL));

	double alpha = 3.0;
	double beta = 9.0;
	double db = 2.25;

	//Brusselator Variables
	double domainScale = 16.0;
	double timeStepRate = 0.1;
	double imW = 128.0;
	double dH = 1;
	double dH_Sq = dH * dH;
	double dT_Org = timeStepRate * dH_Sq;
	double mEL = dH;
	double avL = 0.0;
	float s = 1.0;
	float da = 1.0;

	igl::readOBJ("C:/Users/mmoster/Reaction_Diffusion/Research_RD_Systems/NEW_RD_System/SurfaceGeometry/BunnyClean1000.obj", V, F);

	Eigen::MatrixXd ColorVals(V.rows(), V.cols()), finalColors(V.rows(), V.cols());

	for (int i = 0; i < V.rows(); i++) {
		double u = alpha;
		double v = beta / alpha;
		double noiseL = 0.1*alpha;
		double r = (double(rand())) / double(RAND_MAX);
		u = (u - r * noiseL);

		for (int j = 0; j < V.cols(); j++) {
			if (j == 0) {
				ColorVals(i, j) = u;
			}
			else if (j == 1) {
				ColorVals(i, j) = v;
			}
			else {
				ColorVals(i, j) = 1.00;
			}
		}
	}

	igl::cotmatrix(V, F, L);

	U = V;
	SparseMatrix<double> M;
	igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	const auto & S = (M - 0.001*L);
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
	assert(solver.info() == Eigen::Success);
	U = solver.solve(M*U).eval();

	CV = L * U;
	CV = CV.cwiseProduct(ColorVals);
	for (int j = 0; j < 1000000; j++) {

		if (j % 10000 == 0) {
			cout << j << endl;
		}

		for (int i = 0; i < CV.rows(); i++) {
			double u = ColorVals(i, 0);
			double v = ColorVals(i, 1);
			double lapU = CV(i, 0);
			double lapV = CV(i, 1);

			double Fa = s * (alpha - ((1.0 + beta) * u) + ((u * u) * v));
			double Ga = s * ((beta * u) - ((u * u) * v));
			double delU = Fa + lapU;
			double delV = Ga + (db * lapV);

			u = u + dT_Org * delU;
			v = v + dT_Org * delV;

			u = max(0.0, u);
			v = max(0.0, v);

			ColorVals(i, 0) = u;
			ColorVals(i, 1) = v;
			ColorVals(i, 2) = 1.0;
		}

		double averageU = 0.0;

		for (int q = 0; q < ColorVals.rows(); q++) {
			averageU += ColorVals(q, 0);
		}

		averageU = averageU / ColorVals.rows();

		//cout << averageU << endl;

		double sum = 0.0;
		for (int k = 0; k < ColorVals.rows(); k++) {
			sum += (ColorVals(k, 0) - averageU) * (ColorVals(k, 0) - averageU);
		}

		double sz = ColorVals.rows();
		double stdDev = sqrt((1.0 / (ColorVals.rows() - 1)) * sum);
		double minU = averageU - 3 * stdDev;
		double maxU = averageU + 3 * stdDev;

		for (int r = 0; r < ColorVals.rows(); r++) {
			double currentU = ColorVals(r, 0);
			double u_im = (currentU - minU) / (maxU - minU);

			double red = u_im * 255.0;
			double green = u_im * 255.0;
			double blue = 1.0 * 255.0;

			//cout << red << "\t" << green << "\t" << blue << endl;

			finalColors(r, 0) = red;
			finalColors(r, 1) = green;
			finalColors(r, 2) = blue;
		}

	}

	igl::opengl::glfw::Viewer viewer;
	viewer.data().clear();
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(finalColors);
	viewer.launch();

	//cout << ColorVals << endl;

}
