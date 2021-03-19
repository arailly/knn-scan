#include <iostream>
#include <cpputil.hpp>

using namespace std;
using namespace cpputil;

auto save(const string &path, vector<Neighbors> results) {
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

int main(int argc, char **argv) {
    if (argc != 8) {
        cout << argv[0] << " "
                           "k "
                           "dim "
                           "n "
                           "n_query"
                           "dataset_path "
                           "query_path "
                           "save_path" << endl;
        exit(1);
    }

    int k = atoi(argv[1]);
    int dim = atoi(argv[2]);
    int n = atoi(argv[3]);
    int n_query = atoi(argv[4]);
    string data_path(argv[5]);
    string query_path(argv[6]);
    string save_path(argv[7]);

    auto dataset = DataArray(n, dim);
    dataset.load(data_path);

    auto queries = DataArray(n_query, dim);
    queries.load(query_path);

    vector<Neighbors> results(n_query);

    const auto start = get_now();

#pragma omp parallel for
    for (int query_id = 0; query_id < n_query; ++query_id) {
        const auto query = queries.find(query_id);
        results[query_id] = knn_scan(k, query, dataset);
    }

    const auto stop = get_now();

    cout << "search time: " << get_duration(start, stop) / 1000000
         << "[s]" << endl;

    save(save_path, results);
}