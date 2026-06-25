#pragma once

#include <string>
#include <unordered_map>

#include <Eigen/Geometry>

namespace gmr {

    struct HumanBodyState {
        Eigen::Vector3d position       = Eigen::Vector3d::Zero();
        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();  // wxyz
    };

    using HumanFrame = std::unordered_map<std::string, HumanBodyState>;

}  // namespace gmr
