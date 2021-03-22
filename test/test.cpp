#include <gtest/gtest.h>
#include <cpputil.hpp>

using namespace std;
using namespace cpputil;

TEST(knn_scan, sift) {
    int k = 10, dim = 128, k_max = 100;
    int n = 1000000, n_query = 2;
    string data_path = "/mnt/qnap/data/sift/sift_base.fvecs";
    string query_path = "/mnt/qnap/data/sift/sift_query.fvecs";
    string gt_path = "/mnt/qnap/data/sift/sift_groundtruth.ivecs";

    auto db = DataArray(n, dim);
    db.load(data_path);

    auto queries = DataArray(n_query, dim);
    queries.load(query_path);

    auto gt = GroundTruth(n_query, k_max);
    gt.load(gt_path);

    for (int query_id = 0; query_id < n_query; ++query_id) {
        const auto query = queries.find(query_id);
        const auto result = knn_scan(k, query, db);

        for (int i = 0; i < k; ++i) {
            ASSERT_EQ(result[i].id, gt[query_id][i]);
        }
    }
}
