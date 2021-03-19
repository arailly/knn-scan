//
// Created by Yusuke Arai on 2021/03/19.
//

#ifndef KNN_SCAN_KNN_SCAN_HPP
#define KNN_SCAN_KNN_SCAN_HPP

#include <cpputil.hpp>
#include <map>

using namespace std;
using namespace cpputil;

auto knn_scan(int k, DataArray::Data query, DataArray dataset,
              Dist dist = euclidean_dist) {
    map<float, int> candidates;

    for (int data_id = 0; data_id < dataset.n; ++data_id) {
        const auto data = dataset.find(data_id);
        const auto dist_val = dist(query, data, dataset.dim);

        if (candidates.size() < k) {
            candidates.emplace(dist_val, data_id);
            continue;
        }

        const auto furthest = --candidates.cend();
        if (dist_val < furthest->first) {
            candidates.emplace(dist_val, data_id);
            candidates.erase(furthest);
        }
    }

    Neighbors result;
    for (const auto candidate : candidates) {
        result.emplace_back(candidate.first, candidate.second);
    }

    return result;
}

#endif //KNN_SCAN_KNN_SCAN_HPP
