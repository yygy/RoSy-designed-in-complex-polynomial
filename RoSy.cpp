#include "SDRoSyBuilder.h"

#include <vector>
#include <unordered_set>
#include <map>

#include<Eigen/SparseCholesky>

#include "SystemLibrary/SDLogger.h"
#include "SystemLibrary/SDTimer.h"

#include "Geometry/DebugUtil/DebugUtil.h"
#include "Geometry/SDGeoMesh.h"

SD_NAMESPACE_BEGIN

inline constexpr int USE_EIGEN_ITERATIVE_SOLVER_CRITICAL1 = 1000000;

SDRoSyBuilder::SDRoSyBuilder() {}

//做二维平面的
std::vector<double> SDRoSyBuilder::Run(SDGeoMeshConstPtr geoMesh)
{
	// 边界三角形角度确定
	// 建立内部相邻三角形求导等式
	//std::polar<double>()

	auto getBoundaryVector = [&](const SDVector3i& vertices) -> SDVector {
		const auto& pp = geoMesh->GetPositions();
		for (int i = 0; i < 3; i++)
		{
			int j = GMath::NextIndex(i, 3);
			int edge = geoMesh->FindEdgeBetweenVertices(vertices[i], vertices[j]);
			if (edge >= 0 && geoMesh->IsBoundaryEdge(edge))
			{
				return pp[vertices[j]] - pp[vertices[i]];
			}
		}

		return SDVector::Zero();
	};

	int fSize = geoMesh->GetFacetCount();

	std::vector<int> fixed(fSize, false);
	for (int i = 0; i < fSize; i++)
	{
		fixed[i] = geoMesh->IsBoundaryFacet(i);
	}
	const auto& poss = geoMesh->GetPositions();
	std::vector<Eigen::Triplet<std::complex< double>>> triplets;
	Eigen::Matrix<std::complex<double>, -1, 1> b;
	b.resize(fSize);
	b.setZero();
	for (int i = 0; i < fSize; i++)
	{
		if (fixed[i])
		{
			triplets.emplace_back(Eigen::Triplet<std::complex<double>>(i, i, std::polar<double>(1, 0)));
			const auto& vertices = geoMesh->GetFaceInfos()[i];
			SDVector vec = getBoundaryVector(vertices);
			double alpha = GMath::CrossAngle2D(SDVector::Unit(0), vec);
			b[i] += std::polar<double>(1, 4 * alpha);
		}
		else
		{
			const auto& neibourFaces = geoMesh->FindAdjacentFacetsByFacet(i);
			for (int j : neibourFaces)
			{
				if (geoMesh->FindEdgeBetweenFacets(i, j))
				{
					//通过公共边相邻的
					triplets.emplace_back(Eigen::Triplet<std::complex<double>>(i, i, std::polar<double>(2, 0)));
					triplets.emplace_back(Eigen::Triplet<std::complex<double>>(i, j, -std::polar<double>(2, 0)));
				}
			}
		}
	}

	Eigen::SparseMatrix<std::complex<double>> A;
	A.resize(fSize, fSize);
	A.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::SparseLU<Eigen::SparseMatrix< std::complex<double>>> solver(A);
	Eigen::Matrix<std::complex<double>, -1, 1> result = solver.solve(b);

	std::vector<double> angles(result.size());

	for (int i = 0; i < result.size(); i++)
	{
		std::complex<double>& polyCoeff = result[i];
		float angle = std::atan2(polyCoeff.imag(), polyCoeff.real());
		angles[i] = angle / 4.0f;
	}

	return angles;
}

SD_NAMESPACE_END
