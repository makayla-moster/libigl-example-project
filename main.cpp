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

	Eigen::MatrixXd ColorVals(V.rows(), V.cols());

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

	//cout << ColorVals << endl;

	// Compute Laplace-Beltrami operator: #V by #V
	igl::cotmatrix(V, F, L);

	U = V;
	SparseMatrix<double> M;
	igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	const auto & S = (M - 0.001*L);
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
	assert(solver.info() == Eigen::Success);
	U = solver.solve(M*U).eval();

	//cout << L.rows() << "\t" << L.cols() << endl;
	//cout << U.rows() << "\t" << U.cols() << endl;

	CV = L * U;

	/*cout << ColorVals.rows() << "\t" << ColorVals.cols() << endl;
	cout << CV.rows() << "\t" << CV.cols() << endl;*/
	CV = CV.cwiseProduct(ColorVals);
	//cout << CV.rows() << "\t" << CV.cols() << endl;

	//cout << CV << endl; 

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
	}


	//const auto &key_down = [](igl::opengl::glfw::Viewer &viewer, unsigned char key, int mod)->bool
	//{
	//	switch (key)
	//	{
	//	case 'r':
	//	case 'R':
	//		U = V;
	//		break;
	//	case ' ':
	//	{
	//		// Recompute just mass matrix on each step
	//		SparseMatrix<double> M;
	//		igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	//		// Solve (M-delta*L) U = M*U
	//		const auto & S = (M - 0.001*L);
	//		Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
	//		assert(solver.info() == Eigen::Success);
	//		U = solver.solve(M*U).eval();
	//		// Compute centroid and subtract (also important for numerics)
	//		VectorXd dblA;
	//		igl::doublearea(U, F, dblA);
	//		double area = 0.5*dblA.sum();
	//		MatrixXd BC;
	//		igl::barycenter(U, F, BC);
	//		RowVector3d centroid(0, 0, 0);
	//		for (int i = 0; i < BC.rows(); i++)
	//		{
	//			centroid += 0.5*dblA(i) / area * BC.row(i);
	//		}
	//		U.rowwise() -= centroid;
	//		// Normalize to unit surface area (important for numerics)
	//		U.array() /= sqrt(area);
	//		break;
	//	}
	//	default:
	//		return false;
	//	}
	//	// Send new positions, update normals, recenter
	//	viewer.data().set_vertices(U);
	//	viewer.data().compute_normals();
	//	viewer.core().align_camera_center(U, F);
	//	return true;
	//};


	//// Use original normals as pseudo-colors
	//MatrixXd N;
	//igl::per_vertex_normals(V, F, N);
	//MatrixXd C = N.rowwise().normalized().array()*0.5 + 0.5;

	//// Initialize smoothing with base mesh
	//U = V;
	//viewer.data().set_mesh(U, F);
	//viewer.data().set_colors(C);
	//viewer.callback_key_down = key_down;

	//cout << "Press [space] to smooth." << endl;;
	//cout << "Press [r] to reset." << endl;;
	//return viewer.launch();
}
