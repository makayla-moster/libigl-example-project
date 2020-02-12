#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readDMAT.h>
#include <igl/repdiag.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <igl/min_quad_with_fixed.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/boundary_facets.h>
#include <igl/colon.h>

Eigen::MatrixXd V, U, V2;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L;
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
	
	// compute laplace beltrami operator
	igl::cotmatrix(ColorVals, F, L);

	SparseMatrix<double> M;
	igl::massmatrix(ColorVals, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	// Solve (M-delta*L) U = M*U
	const auto & S = (M - 0.001*L);
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
	assert(solver.info() == Eigen::Success);
	U = solver.solve(M*U).eval();

	cout << U << endl;

}