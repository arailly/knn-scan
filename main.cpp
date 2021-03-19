#include <iostream>
#include <cpputil.hpp>

using namespace std;
using namespace cpputil;

auto save(const string &path, vector <Neighbors> results) {
    ofstream ofs(path);

    if (!ofs) throw runtime_error("invalid path");

    ofs << "query_id,data_id,dist" << endl;

    for (int query_id = 0; query_id < results.size(); ++query_id) {
        for (const auto neighbor : results[query_id]) {
            ofs <<
                query_id << "," <<
                neighbor.id << "," <<
                neighbor.dist << endl;
        }
    }
}

int main() {
    const int n = 1000000, n_query = 10;
    const int dim = 128;
    int k = 5, k_max = 100;

    const string data_path = "/mnt/qnap/data/sift/sift_base.fvecs";
    const string query_path = "/mnt/qnap/data/sift/sift_query.fvecs";
    const string save_path = "/home/arai/workspace/sample.csv";

    auto dataset = DataArray(n, dim);
    dataset.load(data_path);

    auto queries = DataArray(n_query, dim);
    queries.load(query_path);

    vector <Neighbors> results(n_query);

#pragma omp parallel for
    for (int query_id = 0; query_id < n_query; ++query_id) {
        const auto query = queries.find(query_id);
        results[query_id] = knn_scan(k, query, dataset);
    }

    save(save_path, results);
}