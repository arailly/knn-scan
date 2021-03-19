#include <gtest/gtest.h>
#include <cpputil.hpp>
#include <knn-scan.hpp>

using namespace std;
using namespace cpputil;

TEST(knn_scan, euclidean) {
    const int n = 1000000, n_query = 1;
    const int dim = 128;
    int k = 5;

    const string data_path = "/mnt/qnap/data/sift/sift_base.fvecs";
    const string query_path = "/mnt/qnap/data/sift/sift_query.fvecs";

    auto dataset = DataArray(n, dim);
    dataset.load(data_path);

    auto queries = DataArray(n_query, dim);
    queries.load(query_path);

    int query_id = 0;
    const auto query = queries.find(query_id);
    const auto res = knn_scan(k, query, dataset);

    ASSERT_EQ(res.size(), k);
    ASSERT_LT(res[0].dist, 233);
}
