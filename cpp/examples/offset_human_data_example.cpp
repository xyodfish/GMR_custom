#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

struct BodyState {
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();  // wxyz
};

using HumanData = std::unordered_map<std::string, BodyState>;

HumanData offsetHumanData(
    const HumanData& humanData,
    const std::unordered_map<std::string, Eigen::Vector3d>& posOffsets,
    const std::unordered_map<std::string, Eigen::Quaterniond>& rotOffsets) {
    HumanData out;

    for (const auto& [bodyName, body] : humanData) {
        const Eigen::Quaterniond updatedQuat = body.quat * rotOffsets.at(bodyName);
        const Eigen::Vector3d globalPosOffset = updatedQuat * posOffsets.at(bodyName);

        out[bodyName] = BodyState{body.pos + globalPosOffset, updatedQuat};
    }

    return out;
}

void printHumanData(const std::string& title, const HumanData& data, const std::vector<std::string>& order) {
    std::cout << title << "\n";
    for (const std::string& name : order) {
        const BodyState& body = data.at(name);
        std::cout << "  " << name << "\n";
        std::cout << "    pos  = [" << body.pos.x() << ", " << body.pos.y() << ", " << body.pos.z() << "]\n";
        std::cout << "    quat = [" << body.quat.w() << ", " << body.quat.x() << ", " << body.quat.y() << ", "
                  << body.quat.z() << "]\n";
    }
}

int main() {
    constexpr double pi = 3.14159265358979323846;
    const std::vector<std::string> order = {"Pelvis", "LeftHand"};

    HumanData humanData;
    humanData["Pelvis"] = BodyState{Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Quaterniond::Identity()};
    humanData["LeftHand"] =
        BodyState{Eigen::Vector3d(0.3, 0.2, 1.4), Eigen::Quaterniond(Eigen::AngleAxisd(pi / 2.0, Eigen::Vector3d::UnitZ()))};

    std::unordered_map<std::string, Eigen::Vector3d> posOffsets;
    posOffsets["Pelvis"] = Eigen::Vector3d(0.0, 0.0, 0.05);
    posOffsets["LeftHand"] = Eigen::Vector3d(0.1, 0.0, 0.0);

    std::unordered_map<std::string, Eigen::Quaterniond> rotOffsets;
    rotOffsets["Pelvis"] = Eigen::Quaterniond::Identity();
    rotOffsets["LeftHand"] = Eigen::Quaterniond(Eigen::AngleAxisd(pi / 2.0, Eigen::Vector3d::UnitZ()));

    const HumanData out = offsetHumanData(humanData, posOffsets, rotOffsets);

    std::cout << std::fixed << std::setprecision(4);
    printHumanData("before", humanData, order);
    printHumanData("after", out, order);

    return 0;
}
