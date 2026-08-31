#include "gmr/retarget/robot_trajectory_pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <Eigen/Geometry>
#include <Eigen/QR>
#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

namespace gmr {
namespace {

using Contacts = std::vector<std::unordered_map<std::string, bool>>;
using Jacobian = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;

constexpr int kG1Nq = 36;
constexpr double kPi = 3.14159265358979323846;

struct ModelDeleter {
    void operator()(mjModel* model) const {
        mj_deleteModel(model);
    }
};

struct DataDeleter {
    void operator()(mjData* data) const {
        mj_deleteData(data);
    }
};

using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

struct SiteSpec {
    std::string name;
    int bodyId = -1;
    Eigen::Vector3d positionOffset = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientationOffset = Eigen::Quaterniond::Identity();
};

struct CanonicalConfig {
    double height = 1.8;
    Eigen::Vector3d pelvisToSpine3 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pelvisToLeftHip = Eigen::Vector3d::Zero();
    Eigen::Vector3d pelvisToRightHip = Eigen::Vector3d::Zero();
    double thigh = 0.0;
    double shank = 0.0;
    std::vector<std::string> footNames;
    double heightThreshold = 0.04;
    double speedThreshold = 0.35;
    int minContactFrames = 3;
    int smoothWindow = 5;
};

ModelPtr loadModel(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("MuJoCo model not found: " + path.string());
    }

    std::array<char, 1024> error{};
    ModelPtr model(mj_loadXML(path.c_str(), nullptr, error.data(), error.size()));
    if (!model) {
        throw std::runtime_error("Failed to load MuJoCo model '" + path.string() + "': " + error.data());
    }

    return model;
}

Eigen::Vector3d yamlVec3(const YAML::Node& node, const std::string& name) {
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error(name + " must be a three-element sequence.");
    }

    return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

Eigen::Quaterniond yamlQuat(const YAML::Node& node, const std::string& name) {
    if (!node.IsSequence() || node.size() != 4) {
        throw std::runtime_error(name + " must be a four-element wxyz quaternion.");
    }

    Eigen::Quaterniond q(
        node[0].as<double>(),
        node[1].as<double>(),
        node[2].as<double>(),
        node[3].as<double>());
    if (!std::isfinite(q.norm()) || q.norm() < 1e-12) {
        throw std::runtime_error(name + " has a zero or non-finite norm.");
    }

    return q.normalized();
}

std::filesystem::path resolveConfigPath(
    const YAML::Node& cfg,
    const std::string& key,
    const std::filesystem::path& gmrRoot) {
    std::filesystem::path path(cfg[key].as<std::string>());
    if (path.is_relative()) {
        path = gmrRoot / path;
    }

    return std::filesystem::weakly_canonical(path);
}

std::vector<Eigen::VectorXd> parseJsonQpos(const std::filesystem::path& path, double* fpsMetadata = nullptr) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open qpos JSON: " + path.string());
    }

    nlohmann::json root;
    input >> root;
    if (fpsMetadata != nullptr && root.is_object() && root.contains("fps")) {
        *fpsMetadata = root.at("fps").get<double>();
    }

    const nlohmann::json& frames = root.is_object() && root.contains("qpos_frames")
        ? root.at("qpos_frames")
        : root;
    if (!frames.is_array() || frames.empty()) {
        throw std::runtime_error("qpos JSON must contain a non-empty qpos_frames array.");
    }

    std::vector<Eigen::VectorXd> out;
    out.reserve(frames.size());
    for (const auto& row : frames) {
        if (!row.is_array()) {
            throw std::runtime_error("Every qpos frame must be an array.");
        }

        Eigen::VectorXd q(row.size());
        for (std::size_t i = 0; i < row.size(); ++i) {
            q[static_cast<Eigen::Index>(i)] = row[i].get<double>();
        }

        out.push_back(std::move(q));
    }

    return out;
}

std::vector<Eigen::VectorXd> parseCsvQpos(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open trajectory CSV: " + path.string());
    }

    std::vector<Eigen::VectorXd> out;
    std::string line;
    int lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);
        std::string token;
        std::vector<double> values;
        while (std::getline(stream, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid number in CSV line " + std::to_string(lineNumber) + ".");
            }
        }

        if (values.size() != kG1Nq) {
            throw std::runtime_error(
                "G1 CSV line " + std::to_string(lineNumber) + " has " +
                std::to_string(values.size()) + " columns; expected 36.");
        }

        Eigen::VectorXd q(kG1Nq);
        q[0] = values[0];
        q[1] = values[1];
        q[2] = values[2];
        q[3] = values[6];
        q[4] = values[3];
        q[5] = values[4];
        q[6] = values[5];
        for (int i = 7; i < kG1Nq; ++i) {
            q[i] = values[i];
        }

        Eigen::Vector4d quat = q.segment<4>(3);
        const double norm = quat.norm();
        if (!std::isfinite(norm) || norm < 1e-12) {
            throw std::runtime_error("Invalid root quaternion in CSV line " + std::to_string(lineNumber) + ".");
        }

        quat /= norm;
        if (!out.empty() && out.back().segment<4>(3).dot(quat) < 0.0) {
            quat = -quat;
        }

        q.segment<4>(3) = quat;
        out.push_back(std::move(q));
    }

    if (out.empty()) {
        throw std::runtime_error("Trajectory CSV is empty.");
    }

    return out;
}

std::string extractNpyHeaderString(const std::string& header, const std::string& key) {
    const std::size_t keyPos = header.find("'" + key + "'");
    const std::size_t colon = keyPos == std::string::npos ? keyPos : header.find(':', keyPos);
    const std::size_t quote0 = colon == std::string::npos ? colon : header.find('\'', colon);
    const std::size_t quote1 = quote0 == std::string::npos ? quote0 : header.find('\'', quote0 + 1);
    if (quote0 == std::string::npos || quote1 == std::string::npos) {
        throw std::runtime_error("Malformed NPY header field: " + key);
    }

    return header.substr(quote0 + 1, quote1 - quote0 - 1);
}

std::vector<Eigen::VectorXd> parseNpyQpos(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open NPY trajectory: " + path.string());
    }

    std::array<char, 6> magic{};
    input.read(magic.data(), magic.size());
    if (std::memcmp(magic.data(), "\x93NUMPY", magic.size()) != 0) {
        throw std::runtime_error("Invalid NPY magic header.");
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    input.read(reinterpret_cast<char*>(&major), 1);
    input.read(reinterpret_cast<char*>(&minor), 1);
    std::uint32_t headerLength = 0;
    if (major == 1) {
        std::uint16_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        headerLength = value;
    } else if (major == 2 || major == 3) {
        input.read(reinterpret_cast<char*>(&headerLength), sizeof(headerLength));
    } else {
        throw std::runtime_error("Unsupported NPY version " + std::to_string(major) + "." + std::to_string(minor));
    }

    std::string header(headerLength, '\0');
    input.read(header.data(), header.size());
    const std::string descr = extractNpyHeaderString(header, "descr");
    if (header.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error("Fortran-order NPY arrays are not supported.");
    }

    const std::size_t shapePos = header.find("'shape'");
    const std::size_t open = shapePos == std::string::npos ? shapePos : header.find('(', shapePos);
    const std::size_t comma = open == std::string::npos ? open : header.find(',', open);
    const std::size_t close = comma == std::string::npos ? comma : header.find(')', comma);
    if (open == std::string::npos || comma == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("NPY trajectory must have a two-dimensional shape.");
    }

    const std::size_t rows = std::stoull(header.substr(open + 1, comma - open - 1));
    const std::size_t cols = std::stoull(header.substr(comma + 1, close - comma - 1));
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("NPY trajectory shape must be non-empty.");
    }

    const bool isF64 = descr == "<f8" || descr == "=f8";
    const bool isF32 = descr == "<f4" || descr == "=f4";
    if (!isF64 && !isF32) {
        throw std::runtime_error("NPY trajectory dtype must be little-endian float32 or float64, got " + descr + ".");
    }

    std::vector<Eigen::VectorXd> out(rows, Eigen::VectorXd(cols));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (isF64) {
                double value = 0.0;
                input.read(reinterpret_cast<char*>(&value), sizeof(value));
                out[r][static_cast<Eigen::Index>(c)] = value;
            } else {
                float value = 0.0F;
                input.read(reinterpret_cast<char*>(&value), sizeof(value));
                out[r][static_cast<Eigen::Index>(c)] = value;
            }
        }
    }

    if (!input) {
        throw std::runtime_error("NPY payload ended before the declared shape.");
    }

    return out;
}

void validateSourceFrames(const std::vector<Eigen::VectorXd>& frames, int nq) {
    if (frames.empty()) {
        throw std::runtime_error("Source trajectory is empty.");
    }

    for (std::size_t t = 0; t < frames.size(); ++t) {
        const Eigen::VectorXd& q = frames[t];
        if (q.size() != nq || !q.allFinite()) {
            throw std::runtime_error(
                "Source frame " + std::to_string(t) + " must contain " +
                std::to_string(nq) + " finite qpos values.");
        }

        const double quatNorm = q.segment<4>(3).norm();
        if (std::abs(quatNorm - 1.0) > 1e-2) {
            throw std::runtime_error("Source frame " + std::to_string(t) + " has a non-unit root quaternion.");
        }

        if (t > 0 && (q.head<3>() - frames[t - 1].head<3>()).norm() > 0.5) {
            throw std::runtime_error("Source trajectory contains a root translation step above 0.5 m/frame.");
        }
    }
}

Eigen::Quaterniond bodyQuaternion(const mjData* data, int bodyId) {
    const double* q = &data->xquat[4 * bodyId];
    return Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized();
}

Eigen::Vector3d bodyPosition(const mjData* data, int bodyId) {
    return Eigen::Map<const Eigen::Vector3d>(&data->xpos[3 * bodyId]);
}

HumanFrame fitCanonicalFrame(const HumanFrame& target, const CanonicalConfig& cfg);

Eigen::Vector3d clampLength(
    const Eigen::Vector3d& root,
    const Eigen::Vector3d& target,
    double length) {
    const Eigen::Vector3d delta = target - root;
    const double norm = delta.norm();
    if (norm < 1e-8) {
        return root + Eigen::Vector3d(0.0, 0.0, length > 0.0 ? -length : 0.0);
    }

    return root + delta * (length / norm);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> twoBoneIk(
    const Eigen::Vector3d& root,
    const Eigen::Vector3d& target,
    double length1,
    double length2,
    const Eigen::Vector3d& pole) {
    Eigen::Vector3d delta = target - root;
    double distance = delta.norm();
    const double maxReach = length1 + length2 - 1e-4;
    const double minReach = std::abs(length1 - length2) + 1e-4;
    Eigen::Vector3d direction;
    if (distance < 1e-8) {
        direction = Eigen::Vector3d(0.0, 0.0, -1.0);
        distance = minReach;
    } else {
        direction = delta / distance;
    }

    const double reachableDistance = std::clamp(distance, minReach, maxReach);
    const Eigen::Vector3d reachableTarget = root + direction * reachableDistance;
    double cosA = (length1 * length1 + reachableDistance * reachableDistance - length2 * length2) /
        (2.0 * length1 * reachableDistance);
    cosA = std::clamp(cosA, -1.0, 1.0);
    const double sinA = std::sqrt(std::max(0.0, 1.0 - cosA * cosA));

    Eigen::Vector3d poleDirection = pole - root;
    poleDirection -= direction * poleDirection.dot(direction);
    if (poleDirection.norm() < 1e-8) {
        const Eigen::Vector3d helper = std::abs(direction.y()) < 0.9
            ? Eigen::Vector3d(0.0, 1.0, 0.0)
            : Eigen::Vector3d(1.0, 0.0, 0.0);
        poleDirection = direction.cross(helper);
    }

    poleDirection.normalize();
    const Eigen::Vector3d mid = root + direction * (length1 * cosA) + poleDirection * (length1 * sinA);
    return {mid, reachableTarget};
}

int frameCountAtFps(int framesAt30Fps, double fps, int minimum) {
    const int intervalsAt30Fps = std::max(0, framesAt30Fps - 1);
    return std::max(minimum, 1 + static_cast<int>(std::lround(intervalsAt30Fps * fps / 30.0)));
}

int oddWindowAtFps(int framesAt30Fps, double fps) {
    const int oddFramesAt30Fps = std::max(1, framesAt30Fps + (framesAt30Fps % 2 == 0));
    const int radiusAt30Fps = (oddFramesAt30Fps - 1) / 2;
    const int radius = static_cast<int>(std::lround(radiusAt30Fps * fps / 30.0));
    return 2 * radius + 1;
}

CanonicalConfig loadCanonicalConfig(const YAML::Node& cfg, double fps) {
    CanonicalConfig out;
    out.height = cfg["canonical_height_m"].as<double>();
    const YAML::Node bones = cfg["canonical_bones_m"];
    out.pelvisToSpine3 = yamlVec3(bones["pelvis_to_spine3"], "pelvis_to_spine3");
    out.pelvisToLeftHip = yamlVec3(bones["pelvis_to_left_hip"], "pelvis_to_left_hip");
    out.pelvisToRightHip = yamlVec3(bones["pelvis_to_right_hip"], "pelvis_to_right_hip");
    out.thigh = bones["thigh"].as<double>();
    out.shank = bones["shank"].as<double>();

    const YAML::Node contact = cfg["contact"];
    out.footNames = contact["foot_bodies"].as<std::vector<std::string>>();
    out.heightThreshold = contact["height_threshold_m"].as<double>(0.04);
    out.speedThreshold = contact["speed_threshold_mps"].as<double>(0.35);
    out.minContactFrames = frameCountAtFps(contact["min_contact_frames"].as<int>(3), fps, 1);
    out.smoothWindow = oddWindowAtFps(cfg["smoothing"]["window"].as<int>(5), fps);

    return out;
}

Contacts cleanContacts(
    std::unordered_map<std::string, std::vector<bool>> active,
    const CanonicalConfig& cfg,
    int n) {
    for (const std::string& name : cfg.footNames) {
        std::vector<bool>& values = active.at(name);
        int t = 0;
        while (t < n) {
            if (!values[t]) {
                ++t;
                continue;
            }

            const int begin = t;
            while (t < n && values[t]) {
                ++t;
            }

            if (t - begin < cfg.minContactFrames) {
                std::fill(values.begin() + begin, values.begin() + t, false);
            }

        }

        t = 0;
        while (t < n) {
            if (values[t]) {
                ++t;
                continue;
            }

            const int begin = t;
            while (t < n && !values[t]) {
                ++t;
            }

            const bool boundedByContact = begin > 0 && t < n && values[begin - 1] && values[t];
            if (boundedByContact && t - begin < cfg.minContactFrames) {
                std::fill(values.begin() + begin, values.begin() + t, true);
            }

        }

    }

    Contacts contacts(n);
    for (int t = 0; t < n; ++t) {
        for (const std::string& name : cfg.footNames) {
            contacts[t][name] = active.at(name)[t];
        }

    }

    return contacts;
}

Contacts inferContacts(const std::vector<HumanFrame>& frames, const CanonicalConfig& cfg, double fps) {
    const int n = static_cast<int>(frames.size());
    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> positions;
    std::unordered_map<std::string, std::vector<double>> speeds;
    for (const std::string& name : cfg.footNames) {
        auto& values = positions[name];
        values.reserve(n);
        for (const HumanFrame& frame : frames) {
            values.push_back(frame.at(name).position);
        }

        auto& footSpeeds = speeds[name];
        footSpeeds.assign(n, 0.0);
        for (int t = 1; t < n; ++t) {
            footSpeeds[t] = (values[t] - values[t - 1]).norm() * fps;
        }

        if (n >= 2) {
            footSpeeds[0] = footSpeeds[1];
        }
    }

    std::unordered_map<std::string, std::vector<bool>> active;
    const double band = std::max(cfg.heightThreshold, 0.025);
    for (const std::string& name : cfg.footNames) {
        active[name].assign(n, false);
    }

    double groundHeight = std::numeric_limits<double>::infinity();
    for (int t = 0; t < n; ++t) {
        double lower = std::numeric_limits<double>::infinity();
        for (const std::string& name : cfg.footNames) {
            lower = std::min(lower, positions.at(name)[t].z());
        }

        groundHeight = std::min(groundHeight, lower);
    }

    for (int t = 0; t < n; ++t) {
        for (const std::string& name : cfg.footNames) {
            active[name][t] = positions.at(name)[t].z() <= groundHeight + band &&
                speeds.at(name)[t] < cfg.speedThreshold;
        }
    }

    return cleanContacts(std::move(active), cfg, n);
}

std::vector<HumanFrame> smoothFrames(const std::vector<HumanFrame>& frames, int window) {
    if (window <= 1 || frames.size() < 3) {
        return frames;
    }

    const int n = static_cast<int>(frames.size());
    const int half = window / 2;
    std::unordered_map<std::string, std::vector<Eigen::Quaterniond>> quaternions;
    for (const auto& [name, state] : frames.front()) {
        auto& values = quaternions[name];
        values.reserve(n);
        for (const HumanFrame& frame : frames) {
            Eigen::Quaterniond q = frame.at(name).orientation.normalized();
            if (!values.empty() && values.back().coeffs().dot(q.coeffs()) < 0.0) {
                q.coeffs() *= -1.0;
            }

            values.push_back(q);
        }
    }

    std::vector<HumanFrame> out;
    out.reserve(n);
    for (int t = 0; t < n; ++t) {
        const int begin = std::max(0, t - half);
        const int end = std::min(n, t + half + 1);
        HumanFrame frame;
        for (const auto& [name, unused] : frames.front()) {
            HumanBodyState state;
            Eigen::Vector4d quaternion = Eigen::Vector4d::Zero();
            for (int k = begin; k < end; ++k) {
                state.position += frames[k].at(name).position;
                const Eigen::Quaterniond& q = quaternions.at(name)[k];
                quaternion += Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
            }

            state.position /= static_cast<double>(end - begin);
            quaternion /= static_cast<double>(end - begin);
            Eigen::Quaterniond q(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
            state.orientation = q.normalized();
            if (!out.empty() && out.back().at(name).orientation.coeffs().dot(state.orientation.coeffs()) < 0.0) {
                state.orientation.coeffs() *= -1.0;
            }

            frame[name] = state;
        }

        out.push_back(std::move(frame));
    }

    return out;
}

HumanFrame fitCanonicalFrame(const HumanFrame& target, const CanonicalConfig& cfg) {
    HumanFrame out;
    out["pelvis"] = target.at("pelvis");
    out["spine3"] = target.at("spine3");
    out["spine3"].position = clampLength(
        out.at("pelvis").position,
        target.at("spine3").position,
        cfg.pelvisToSpine3.norm());

    out["left_hip"] = target.at("left_hip");
    out["right_hip"] = target.at("right_hip");
    out["left_hip"].position = clampLength(
        out.at("pelvis").position,
        target.at("left_hip").position,
        cfg.pelvisToLeftHip.norm());
    out["right_hip"].position = clampLength(
        out.at("pelvis").position,
        target.at("right_hip").position,
        cfg.pelvisToRightHip.norm());

    const auto leftLeg = twoBoneIk(
        out.at("left_hip").position,
        target.at("left_foot").position,
        cfg.thigh,
        cfg.shank,
        target.at("left_knee").position);
    const auto rightLeg = twoBoneIk(
        out.at("right_hip").position,
        target.at("right_foot").position,
        cfg.thigh,
        cfg.shank,
        target.at("right_knee").position);
    out["left_knee"] = target.at("left_knee");
    out["right_knee"] = target.at("right_knee");
    out["left_foot"] = target.at("left_foot");
    out["right_foot"] = target.at("right_foot");
    out["left_knee"].position = leftLeg.first;
    out["left_foot"].position = leftLeg.second;
    out["right_knee"].position = rightLeg.first;
    out["right_foot"].position = rightLeg.second;

    for (const std::string& name : {
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}) {
        out[name] = target.at(name);
    }

    const auto head = target.find("head");
    if (head != target.end()) {
        out["head"] = head->second;
    }

    return out;
}

double percentile95(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double index = 0.95 * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    const double alpha = index - static_cast<double>(lower);
    return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

CanonicalFitQuality measureQuality(
    const std::vector<HumanFrame>& targets,
    const std::vector<HumanFrame>& fitted,
    const Contacts& contacts,
    double fps) {
    const std::array<std::string, 5> keyBodies = {
        "pelvis", "left_foot", "right_foot", "left_wrist", "right_wrist"};
    std::vector<double> errors;
    std::vector<double> rotationErrors;
    for (std::size_t t = 0; t < targets.size(); ++t) {
        for (const std::string& name : keyBodies) {
            errors.push_back((targets[t].at(name).position - fitted[t].at(name).position).norm());
            const double dot = std::abs(
                targets[t].at(name).orientation.normalized().dot(fitted[t].at(name).orientation.normalized()));
            rotationErrors.push_back(2.0 * std::acos(std::clamp(dot, 0.0, 1.0)) * 180.0 / kPi);
        }
    }

    std::vector<double> slips;
    for (const std::string& name : {"left_foot", "right_foot"}) {
        for (std::size_t t = 1; t < fitted.size(); ++t) {
            if (contacts[t].at(name) && contacts[t - 1].at(name)) {
                slips.push_back((fitted[t].at(name).position - fitted[t - 1].at(name).position).head<2>().norm() * fps);
            }
        }
    }

    CanonicalFitQuality quality;
    quality.semanticPositionRmseM = std::sqrt(
        std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0) /
        static_cast<double>(std::max<std::size_t>(1, errors.size())));
    quality.semanticPositionP95M = percentile95(errors);
    quality.semanticRotationMeanDeg = rotationErrors.empty()
        ? 0.0
        : std::accumulate(rotationErrors.begin(), rotationErrors.end(), 0.0) / rotationErrors.size();
    quality.contactSlipMeanMps = slips.empty()
        ? 0.0
        : std::accumulate(slips.begin(), slips.end(), 0.0) / slips.size();
    return quality;
}

void groundAlignFrames(std::vector<HumanFrame>& frames) {
    double lower = std::numeric_limits<double>::infinity();
    for (const HumanFrame& frame : frames) {
        lower = std::min(lower, frame.at("left_foot").position.z());
        lower = std::min(lower, frame.at("right_foot").position.z());
    }

    if (!std::isfinite(lower) || std::abs(lower) < 1e-6) {
        return;
    }

    for (HumanFrame& frame : frames) {
        for (auto& [name, state] : frame) {
            state.position.z() -= lower;
        }
    }
}

bool sideMatches(const std::string& text, const std::string& side) {
    const bool left = text.find("left") != std::string::npos || text.find("leg_l") != std::string::npos ||
        text.rfind("l_", 0) == 0 || text.find("_l_") != std::string::npos || text.find("ankle_l") != std::string::npos ||
        text.find("foot_l") != std::string::npos || text.find("toe_l") != std::string::npos;
    const bool right = text.find("right") != std::string::npos || text.find("leg_r") != std::string::npos ||
        text.rfind("r_", 0) == 0 || text.find("_r_") != std::string::npos || text.find("ankle_r") != std::string::npos ||
        text.find("foot_r") != std::string::npos || text.find("toe_r") != std::string::npos;
    return side == "left" ? left : right;
}

struct FootReference {
    int body = -1;
    int site = -1;
};

Eigen::Vector3d footReferencePosition(const mjData* data, const FootReference& foot) {
    const double* position = foot.site >= 0
        ? &data->site_xpos[3 * foot.site]
        : &data->xpos[3 * foot.body];
    return Eigen::Vector3d(position[0], position[1], position[2]);
}

std::unordered_map<std::string, FootReference> resolveFootReferences(const mjModel* model) {
    const std::unordered_map<std::string, std::vector<std::string>> candidates = {
        {"left_foot", {"left_sole_link", "left_foot_link", "left_foot", "LeftFoot",
                       "left_toe_link", "toeLeft", "left_foot_roll_link", "left_ankle_roll_link",
                       "left_foot_pitch_link", "left_ankle_pitch_link", "leg_left_ankle_roll",
                       "leg_left_ankle_pitch", "l_ankle_roll_link", "l_ankle_pitch_link",
                       "ankle_roll_l_link", "ankle_pitch_l_link", "anklePitchLeft", "leg_l6_link",
                       "left_ankle_link"}},
        {"right_foot", {"right_sole_link", "right_foot_link", "right_foot", "RightFoot",
                        "right_toe_link", "toeRight", "right_foot_roll_link", "right_ankle_roll_link",
                        "right_foot_pitch_link", "right_ankle_pitch_link", "leg_right_ankle_roll",
                        "leg_right_ankle_pitch", "r_ankle_roll_link", "r_ankle_pitch_link",
                        "ankle_roll_r_link", "ankle_pitch_r_link", "anklePitchRight", "leg_r6_link",
                        "right_ankle_link"}},
    };

    std::unordered_map<std::string, FootReference> out;
    for (const auto& [key, names] : candidates) {
        const int site = mj_name2id(model, mjOBJ_SITE, key.c_str());
        if (site >= 0) {
            out[key] = FootReference{model->site_bodyid[site], site};
            continue;
        }

        for (const std::string& name : names) {
            const int id = mj_name2id(model, mjOBJ_BODY, name.c_str());
            if (id >= 0) {
                out[key] = FootReference{id, -1};
                break;
            }
        }

        if (out.count(key) != 0) {
            continue;
        }

        const std::string side = key == "left_foot" ? "left" : "right";
        for (int id = 0; id < model->nbody; ++id) {
            const char* raw = mj_id2name(model, mjOBJ_BODY, id);
            const std::string name = raw == nullptr ? "" : raw;
            if (sideMatches(name, side) &&
                (name.find("foot") != std::string::npos || name.find("ankle") != std::string::npos ||
                 name.find("sole") != std::string::npos || name.find("toe") != std::string::npos)) {
                out[key] = FootReference{id, -1};
                break;
            }
        }

        if (out.count(key) == 0) {
            throw std::runtime_error("Could not resolve target robot body for " + key + ".");
        }
    }

    return out;
}

std::vector<int> resolveAnkleJoints(const mjModel* model, const std::string& side) {
    std::vector<int> out;
    for (int joint = 0; joint < model->njnt; ++joint) {
        if (model->jnt_type[joint] != mjJNT_HINGE) {
            continue;
        }

        const char* raw = mj_id2name(model, mjOBJ_JOINT, joint);
        const std::string name = raw == nullptr ? "" : raw;
        if (name.find(side + "_ankle") != std::string::npos ||
            name.find(side + "_foot") != std::string::npos) {
            out.push_back(joint);
        }
    }

    return out;
}

void clipHinge(const mjModel* model, mjData* data, int joint) {
    if (model->jnt_limited[joint]) {
        const int qadr = model->jnt_qposadr[joint];
        data->qpos[qadr] = std::clamp(data->qpos[qadr], model->jnt_range[2 * joint], model->jnt_range[2 * joint + 1]);
    }
}

double measureStanceSlip(
    const mjModel* model,
    mjData* data,
    const std::vector<Eigen::VectorXd>& qpos,
    const Contacts& contacts,
    double fps,
    const std::unordered_map<std::string, FootReference>& feet) {
    std::unordered_map<std::string, Eigen::Vector2d> previous;
    std::vector<double> slips;
    for (std::size_t t = 0; t < qpos.size(); ++t) {
        mju_copy(data->qpos, qpos[t].data(), model->nq);
        mj_forward(model, data);
        for (const auto& [name, foot] : feet) {
            const Eigen::Vector2d xy = footReferencePosition(data, foot).head<2>();
            if (t > 0 && contacts[t].at(name) && contacts[t - 1].at(name) && previous.count(name) != 0) {
                slips.push_back((xy - previous.at(name)).norm() * fps);
            }

            if (contacts[t].at(name)) {
                previous[name] = xy;
            } else {
                previous.erase(name);
            }
        }
    }

    return slips.empty() ? 0.0 : std::accumulate(slips.begin(), slips.end(), 0.0) / slips.size();
}

void plantStanceFeet(
    const mjModel* model,
    mjData* data,
    std::vector<Eigen::VectorXd>& qpos,
    const Contacts& contacts,
    const std::unordered_map<std::string, FootReference>& feet) {
    std::vector<int> joints;
    std::vector<int> qIndices;
    std::vector<int> vIndices;
    for (int joint = 0; joint < model->njnt; ++joint) {
        if (model->jnt_type[joint] != mjJNT_HINGE) {
            continue;
        }

        const char* raw = mj_id2name(model, mjOBJ_JOINT, joint);
        const std::string name = raw == nullptr ? "" : raw;
        if (name.find("hip") == std::string::npos && name.find("knee") == std::string::npos &&
            name.find("ankle") == std::string::npos) {
            continue;
        }

        joints.push_back(joint);
        qIndices.push_back(model->jnt_qposadr[joint]);
        vIndices.push_back(model->jnt_dofadr[joint]);
    }

    if (joints.empty()) {
        return;
    }

    std::unordered_map<std::string, Eigen::Vector2d> hold;
    Jacobian jacp(3, model->nv);
    Jacobian jacr(3, model->nv);
    for (std::size_t t = 0; t < qpos.size(); ++t) {
        for (const auto& [name, foot] : feet) {
            if (!contacts[t].at(name)) {
                hold.erase(name);
            }
        }

        mju_copy(data->qpos, qpos[t].data(), model->nq);
        mj_forward(model, data);
        for (const auto& [name, foot] : feet) {
            if (contacts[t].at(name) && hold.count(name) == 0) {
                hold[name] = footReferencePosition(data, foot).head<2>();
            }
        }

        for (int iteration = 0; iteration < 8; ++iteration) {
            mj_forward(model, data);
            Eigen::VectorXd delta = Eigen::VectorXd::Zero(model->nv);
            int active = 0;
            for (const auto& [name, foot] : feet) {
                if (!contacts[t].at(name) || hold.count(name) == 0) {
                    continue;
                }

                const Eigen::Vector3d position = footReferencePosition(data, foot);
                mj_jac(model, data, jacp.data(), jacr.data(), position.data(), foot.body);
                const Eigen::Vector2d error = hold.at(name) - position.head<2>();
                if (error.norm() < 1e-4) {
                    continue;
                }

                Eigen::MatrixXd j(2, vIndices.size());
                for (std::size_t i = 0; i < vIndices.size(); ++i) {
                    j.col(i) = jacp.block<2, 1>(0, vIndices[i]);
                }

                const Eigen::VectorXd dq = j.completeOrthogonalDecomposition().solve(error);
                for (std::size_t i = 0; i < vIndices.size(); ++i) {
                    delta[vIndices[i]] += dq[i];
                }

                ++active;
            }

            if (active == 0 || delta.norm() < 1e-6) {
                break;
            }

            for (std::size_t i = 0; i < joints.size(); ++i) {
                data->qpos[qIndices[i]] += 0.6 * delta[vIndices[i]];
                clipHinge(model, data, joints[i]);
            }
        }

        qpos[t] = Eigen::Map<const Eigen::VectorXd>(data->qpos, model->nq);
    }
}

void flattenStanceFeet(
    const mjModel* model,
    mjData* data,
    std::vector<Eigen::VectorXd>& qpos,
    const Contacts& contacts,
    const std::unordered_map<std::string, FootReference>& feet,
    double fps) {
    std::unordered_map<std::string, std::vector<int>> ankleJoints = {
        {"left_foot", resolveAnkleJoints(model, "left")},
        {"right_foot", resolveAnkleJoints(model, "right")},
    };
    std::vector<Eigen::VectorXd> desired = qpos;
    Jacobian jacr(3, model->nv);
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    for (std::size_t t = 0; t < qpos.size(); ++t) {
        mju_copy(data->qpos, desired[t].data(), model->nq);
        for (int iteration = 0; iteration < 10; ++iteration) {
            mj_forward(model, data);
            bool moved = false;
            for (const auto& [name, foot] : feet) {
                const std::vector<int>& joints = ankleJoints.at(name);
                if (!contacts[t].at(name) || joints.empty()) {
                    continue;
                }

                const double* matrix = &data->xmat[9 * foot.body];
                const Eigen::Vector3d footZ(matrix[2], matrix[5], matrix[8]);
                const Eigen::Vector3d error = footZ.cross(up);
                if (error.norm() < 1e-4) {
                    continue;
                }

                mj_jacBody(model, data, nullptr, jacr.data(), foot.body);
                Eigen::MatrixXd j(3, joints.size());
                for (std::size_t i = 0; i < joints.size(); ++i) {
                    j.col(i) = jacr.col(model->jnt_dofadr[joints[i]]);
                }

                const Eigen::VectorXd dq = j.completeOrthogonalDecomposition().solve(error);
                for (std::size_t i = 0; i < joints.size(); ++i) {
                    data->qpos[model->jnt_qposadr[joints[i]]] += 0.7 * dq[i];
                    clipHinge(model, data, joints[i]);
                }

                moved = true;
            }

            if (!moved) {
                break;
            }
        }

        desired[t] = Eigen::Map<const Eigen::VectorXd>(data->qpos, model->nq);
    }

    const int half = std::max(1, static_cast<int>(std::lround(4.0 * fps / 30.0)));
    for (const auto& [name, joints] : ankleJoints) {
        std::vector<int> contactFrames;
        for (std::size_t t = 0; t < contacts.size(); ++t) {
            if (contacts[t].at(name)) {
                contactFrames.push_back(static_cast<int>(t));
            }
        }

        if (contactFrames.empty()) {
            continue;
        }

        for (int joint : joints) {
            const int qadr = model->jnt_qposadr[joint];
            std::vector<double> smoothed(qpos.size(), 0.0);
            int begin = 0;
            while (begin < static_cast<int>(qpos.size())) {
                while (begin < static_cast<int>(qpos.size()) && !contacts[begin].at(name)) {
                    ++begin;
                }

                if (begin == static_cast<int>(qpos.size())) {
                    break;
                }

                int end = begin;
                while (end + 1 < static_cast<int>(qpos.size()) && contacts[end + 1].at(name)) {
                    ++end;
                }

                for (int t = begin; t <= end; ++t) {
                    double sum = 0.0;
                    int count = 0;
                    for (int index = std::max(begin, t - half); index <= std::min(end, t + half); ++index) {
                        sum += desired[index][qadr] - qpos[index][qadr];
                        ++count;
                    }

                    smoothed[t] = sum / static_cast<double>(count);
                }

                begin = end + 1;
            }

            for (std::size_t t = 0; t < qpos.size(); ++t) {
                if (!contacts[t].at(name)) {
                    continue;
                }

                qpos[t][qadr] += smoothed[t];
                if (model->jnt_limited[joint]) {
                    qpos[t][qadr] = std::clamp(
                        qpos[t][qadr], model->jnt_range[2 * joint], model->jnt_range[2 * joint + 1]);
                }
            }
        }

    }

}

std::unordered_map<std::string, std::vector<int>> footCollisionGeoms(const mjModel* model) {
    std::unordered_map<std::string, std::vector<int>> ids = {
        {"left_foot", {}},
        {"right_foot", {}},
    };
    for (int geom = 0; geom < model->ngeom; ++geom) {
        if (model->geom_type[geom] == mjGEOM_PLANE || model->geom_contype[geom] == 0) {
            continue;
        }

        const char* geomRaw = mj_id2name(model, mjOBJ_GEOM, geom);
        const char* bodyRaw = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[geom]);
        const std::string text = std::string(geomRaw == nullptr ? "" : geomRaw) + " " +
            std::string(bodyRaw == nullptr ? "" : bodyRaw);
        const bool isFoot = text.find("foot") != std::string::npos || text.find("toe") != std::string::npos ||
            text.find("sole") != std::string::npos || text.find("ankle") != std::string::npos;
        if (!isFoot) {
            continue;
        }

        if (text.find("left") != std::string::npos) {
            ids["left_foot"].push_back(geom);
        } else if (text.find("right") != std::string::npos) {
            ids["right_foot"].push_back(geom);
        }
    }

    return ids;
}

double geomSoleZ(const mjModel* model, const mjData* data, int geom) {
    const Eigen::Vector3d position = Eigen::Map<const Eigen::Vector3d>(&data->geom_xpos[3 * geom]);
    const double* matrix = &data->geom_xmat[9 * geom];
    const Eigen::Vector3d axis(matrix[2], matrix[5], matrix[8]);
    const double* size = &model->geom_size[3 * geom];
    switch (model->geom_type[geom]) {
        case mjGEOM_SPHERE:
            return position.z() - size[0];
        case mjGEOM_CAPSULE:
            return position.z() - std::abs(axis.z()) * size[1] - size[0];
        case mjGEOM_CYLINDER: {
            const double radial = size[0] * std::sqrt(std::max(0.0, 1.0 - axis.z() * axis.z()));
            return position.z() - std::abs(axis.z()) * size[1] - radial;
        }
        case mjGEOM_BOX: {
            double projection = 0.0;
            for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
                projection += std::abs(matrix[6 + axisIndex]) * size[axisIndex];
            }

            return position.z() - projection;
        }
        case mjGEOM_MESH: {
            const int mesh = model->geom_dataid[geom];
            const int begin = model->mesh_vertadr[mesh];
            const int count = model->mesh_vertnum[mesh];
            double lower = std::numeric_limits<double>::infinity();
            for (int i = 0; i < count; ++i) {
                const float* vertex = &model->mesh_vert[3 * (begin + i)];
                const double z = position.z() + matrix[6] * vertex[0] + matrix[7] * vertex[1] + matrix[8] * vertex[2];
                lower = std::min(lower, z);
            }

            return lower;
        }
        default:
            return position.z() - size[0];
    }
}

Contacts inferSourceRobotContacts(
    const mjModel* model,
    mjData* data,
    const std::vector<Eigen::VectorXd>& qpos,
    const CanonicalConfig& cfg) {
    constexpr double kContactMargin = 0.02;
    const int floorGeom = mj_name2id(model, mjOBJ_GEOM, "floor");
    const auto footGeoms = footCollisionGeoms(model);
    if (floorGeom < 0) {
        throw std::runtime_error("Source robot model is missing the floor geometry required for contact inference.");
    }

    std::unordered_map<std::string, std::vector<bool>> active;
    for (const std::string& name : cfg.footNames) {
        if (footGeoms.at(name).empty()) {
            throw std::runtime_error("Source robot model has no collision geometry for " + name + ".");
        }

        active[name].resize(qpos.size());
    }

    for (std::size_t t = 0; t < qpos.size(); ++t) {
        mju_copy(data->qpos, qpos[t].data(), model->nq);
        mj_forward(model, data);
        const double floorZ = data->geom_xpos[3 * floorGeom + 2];
        for (const std::string& name : cfg.footNames) {
            double soleZ = std::numeric_limits<double>::infinity();
            for (int geom : footGeoms.at(name)) {
                soleZ = std::min(soleZ, geomSoleZ(model, data, geom));
            }

            active[name][t] = soleZ <= floorZ + kContactMargin;
        }

    }

    return cleanContacts(std::move(active), cfg, static_cast<int>(qpos.size()));
}

void snapToGround(
    const mjModel* model,
    mjData* data,
    std::vector<Eigen::VectorXd>& qpos,
    const Contacts& contacts,
    const std::unordered_map<std::string, FootReference>& feet) {
    if (model->nq < 7 || model->jnt_type[0] != mjJNT_FREE) {
        return;
    }

    const auto geoms = footCollisionGeoms(model);
    std::unordered_map<std::string, std::vector<double>> soleZByFoot;
    for (const auto& [name, foot] : feet) {
        soleZByFoot[name].resize(qpos.size());
    }

    std::vector<double> lowestSoleZ(qpos.size());
    for (std::size_t t = 0; t < qpos.size(); ++t) {
        mju_copy(data->qpos, qpos[t].data(), model->nq);
        mj_forward(model, data);
        double lower = std::numeric_limits<double>::infinity();
        for (const auto& [name, foot] : feet) {
            double footLower = std::numeric_limits<double>::infinity();
            const auto geomIt = geoms.find(name);
            if (geomIt != geoms.end() && !geomIt->second.empty()) {
                for (int geom : geomIt->second) {
                    footLower = std::min(footLower, geomSoleZ(model, data, geom));
                }
            } else {
                footLower = footReferencePosition(data, foot).z();
            }

            soleZByFoot[name][t] = footLower;
            lower = std::min(lower, footLower);
        }

        lowestSoleZ[t] = lower;
    }

    int floorGeom = mj_name2id(model, mjOBJ_GEOM, "floor");
    if (floorGeom < 0) {
        for (int geom = 0; geom < model->ngeom; ++geom) {
            if (model->geom_type[geom] == mjGEOM_PLANE) {
                floorGeom = geom;
                break;
            }
        }
    }

    const double floorZ = floorGeom >= 0 ? data->geom_xpos[3 * floorGeom + 2] : 0.0;
    for (std::size_t t = 0; t < contacts.size(); ++t) {
        double supportLower = std::numeric_limits<double>::infinity();
        for (const auto& [name, values] : soleZByFoot) {
            if (contacts[t].at(name)) {
                supportLower = std::min(supportLower, values[t]);
            }
        }

        if (!std::isfinite(supportLower)) {
            continue;
        }

        // Root-only grounding cannot lower a support foot through a swing foot that is
        // already closer to the floor. The remaining support error belongs to leg IK.
        const double correction = supportLower - floorZ;
        const double lowestClearance = lowestSoleZ[t] - floorZ;
        qpos[t][2] -= std::min(correction, lowestClearance);
    }
}

void alignWrists(const mjModel* model, mjData* data, std::vector<Eigen::VectorXd>& qpos) {
    struct WristIds {
        int pitch = -1;
        int yaw = -1;
        int elbowBody = -1;
        int rollBody = -1;
        int yawBody = -1;
    };

    std::vector<WristIds> wrists;
    for (const std::string& side : {"left", "right"}) {
        WristIds ids;
        ids.pitch = mj_name2id(model, mjOBJ_JOINT, (side + "_wrist_pitch_joint").c_str());
        ids.yaw = mj_name2id(model, mjOBJ_JOINT, (side + "_wrist_yaw_joint").c_str());
        ids.elbowBody = mj_name2id(model, mjOBJ_BODY, (side + "_elbow_link").c_str());
        ids.rollBody = mj_name2id(model, mjOBJ_BODY, (side + "_wrist_roll_link").c_str());
        ids.yawBody = mj_name2id(model, mjOBJ_BODY, (side + "_wrist_yaw_link").c_str());
        if (ids.pitch < 0 || ids.yaw < 0 || ids.elbowBody < 0 || ids.rollBody < 0 || ids.yawBody < 0) {
            return;
        }

        wrists.push_back(ids);
    }

    Jacobian jacr(3, model->nv);
    for (Eigen::VectorXd& q : qpos) {
        mju_copy(data->qpos, q.data(), model->nq);
        for (int iteration = 0; iteration < 12; ++iteration) {
            mj_forward(model, data);
            bool moved = false;
            for (const WristIds& ids : wrists) {
                Eigen::Vector3d forearm = bodyPosition(data, ids.rollBody) - bodyPosition(data, ids.elbowBody);
                if (forearm.norm() < 1e-8) {
                    continue;
                }

                forearm.normalize();
                const double* matrix = &data->xmat[9 * ids.yawBody];
                const Eigen::Vector3d handX(matrix[0], matrix[3], matrix[6]);
                if (handX.cross(forearm).norm() < 1e-4) {
                    continue;
                }

                mj_jacBody(model, data, nullptr, jacr.data(), ids.yawBody);
                Eigen::Matrix<double, 3, 2> jw;
                jw.col(0) = jacr.col(model->jnt_dofadr[ids.pitch]);
                jw.col(1) = jacr.col(model->jnt_dofadr[ids.yaw]);
                Eigen::Matrix3d skew;
                skew << 0.0, -handX.z(), handX.y(),
                    handX.z(), 0.0, -handX.x(),
                    -handX.y(), handX.x(), 0.0;
                const Eigen::Matrix<double, 3, 2> jDirection = -skew * jw;
                const Eigen::Vector2d dq = jDirection.completeOrthogonalDecomposition().solve(forearm - handX);
                data->qpos[model->jnt_qposadr[ids.pitch]] += 0.8 * dq[0];
                data->qpos[model->jnt_qposadr[ids.yaw]] += 0.8 * dq[1];
                clipHinge(model, data, ids.pitch);
                clipHinge(model, data, ids.yaw);
                moved = true;
            }

            if (!moved) {
                break;
            }
        }

        q = Eigen::Map<const Eigen::VectorXd>(data->qpos, model->nq);
    }
}

nlohmann::json bodyStateJson(const HumanBodyState& state) {
    return {
        {"position", {state.position.x(), state.position.y(), state.position.z()}},
        {"orientation", {state.orientation.w(), state.orientation.x(), state.orientation.y(), state.orientation.z()}},
    };
}

}  // namespace

struct SourceRobotFrameMapper::Impl {
    ModelPtr model;
    DataPtr data;
    std::vector<SiteSpec> sites;
    double outputHeight = 1.8;
    double scale = 1.0;
    bool scaleRootXy = false;
    bool rootInitialized = false;
    Eigen::Vector3d root0 = Eigen::Vector3d::Zero();
    std::unordered_map<std::string, Eigen::Quaterniond> previousQuaternions;
    std::unordered_map<std::string, std::vector<int>> footGeoms;
    std::unordered_map<std::string, bool> footContacts;
    int floorGeom = -1;
};

SourceRobotFrameMapper::SourceRobotFrameMapper(
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& gmrRoot)
    : impl_(std::make_unique<Impl>()) {
    if (!std::filesystem::is_regular_file(mappingPath)) {
        throw std::runtime_error("Source mapping not found: " + mappingPath.string());
    }

    const YAML::Node yaml = YAML::LoadFile(mappingPath.string());
    impl_->model = loadModel(resolveConfigPath(yaml, "robot_model", gmrRoot));
    impl_->data.reset(mj_makeData(impl_->model.get()));
    if (!impl_->data) {
        throw std::runtime_error("Failed to allocate source MuJoCo data.");
    }

    impl_->outputHeight = yaml["canonical_height_m"].as<double>();
    const double sourceHeight = yaml["source_robot_reference_height_m"].as<double>();
    if (!std::isfinite(impl_->outputHeight) || impl_->outputHeight <= 0.0 ||
        !std::isfinite(sourceHeight) || sourceHeight <= 0.0) {
        throw std::runtime_error("Source and output reference heights must be positive and finite.");
    }

    impl_->scale = impl_->outputHeight / sourceHeight;
    impl_->scaleRootXy = yaml["scale_root_xy"].as<bool>(false);
    for (auto it = yaml["sites"].begin(); it != yaml["sites"].end(); ++it) {
        SiteSpec site;
        site.name = it->first.as<std::string>();
        const YAML::Node spec = it->second;
        const std::string bodyName = spec["source_body"].as<std::string>();
        site.bodyId = mj_name2id(impl_->model.get(), mjOBJ_BODY, bodyName.c_str());
        if (site.bodyId < 0) {
            throw std::runtime_error(
                "Unknown source body '" + bodyName + "' for site '" + site.name + "'.");
        }

        site.positionOffset = yamlVec3(
            spec["position_offset_local"], site.name + ".position_offset_local");
        site.orientationOffset = yamlQuat(
            spec["orientation_offset_wxyz"], site.name + ".orientation_offset_wxyz");
        impl_->sites.push_back(std::move(site));
    }

    if (std::none_of(
            impl_->sites.begin(),
            impl_->sites.end(),
            [](const SiteSpec& site) { return site.name == "pelvis"; })) {
        throw std::runtime_error("Source mapping is missing required semantic site: pelvis");
    }

    impl_->floorGeom = mj_name2id(impl_->model.get(), mjOBJ_GEOM, "floor");
    impl_->footGeoms = footCollisionGeoms(impl_->model.get());
    if (impl_->floorGeom < 0) {
        throw std::runtime_error("Source robot model is missing the floor geometry required for contact inference.");
    }

    for (const std::string& name : yaml["contact"]["foot_bodies"].as<std::vector<std::string>>()) {
        if (impl_->footGeoms.at(name).empty()) {
            throw std::runtime_error("Source robot model has no collision geometry for " + name + ".");
        }

        impl_->footContacts[name] = false;
    }
}

SourceRobotFrameMapper::~SourceRobotFrameMapper() = default;

HumanFrame SourceRobotFrameMapper::mapFrame(const Eigen::VectorXd& qpos) {
    if (qpos.size() != impl_->model->nq || !qpos.allFinite()) {
        throw std::runtime_error("Live source qpos must match source model.nq and contain finite values.");
    }

    mju_copy(impl_->data->qpos, qpos.data(), impl_->model->nq);
    mj_forward(impl_->model.get(), impl_->data.get());

    constexpr double kContactMargin = 0.02;
    const double floorZ = impl_->data->geom_xpos[3 * impl_->floorGeom + 2];
    for (auto& [name, contact] : impl_->footContacts) {
        double soleZ = std::numeric_limits<double>::infinity();
        for (int geom : impl_->footGeoms.at(name)) {
            soleZ = std::min(soleZ, geomSoleZ(impl_->model.get(), impl_->data.get(), geom));
        }

        contact = soleZ <= floorZ + kContactMargin;
    }

    HumanFrame raw;
    for (const SiteSpec& site : impl_->sites) {
        const Eigen::Quaterniond linkOrientation = bodyQuaternion(impl_->data.get(), site.bodyId);
        HumanBodyState state;
        state.position = bodyPosition(impl_->data.get(), site.bodyId) +
            linkOrientation * site.positionOffset;
        state.orientation = (linkOrientation * site.orientationOffset).normalized();
        raw[site.name] = state;
    }

    const Eigen::Vector3d pelvis = raw.at("pelvis").position;
    if (!impl_->rootInitialized) {
        impl_->root0 = pelvis;
        impl_->rootInitialized = true;
    }

    Eigen::Vector3d scaledPelvis = pelvis;
    scaledPelvis.z() = impl_->root0.z() * impl_->scale +
        (pelvis.z() - impl_->root0.z()) * impl_->scale;
    if (impl_->scaleRootXy) {
        scaledPelvis.head<2>() = impl_->root0.head<2>() +
            (pelvis - impl_->root0).head<2>() * impl_->scale;
    }

    HumanFrame scaled;
    for (auto& [name, state] : raw) {
        state.position = scaledPelvis + (state.position - pelvis) * impl_->scale;
        const auto previous = impl_->previousQuaternions.find(name);
        if (previous != impl_->previousQuaternions.end() &&
            previous->second.coeffs().dot(state.orientation.coeffs()) < 0.0) {
            state.orientation.coeffs() *= -1.0;
        }

        impl_->previousQuaternions[name] = state.orientation;
        scaled[name] = state;
    }

    return scaled;
}

const std::unordered_map<std::string, bool>& SourceRobotFrameMapper::footContacts() const {
    return impl_->footContacts;
}

void SourceRobotFrameMapper::reset() {
    impl_->rootInitialized = false;
    impl_->root0.setZero();
    impl_->previousQuaternions.clear();
    for (auto& [name, contact] : impl_->footContacts) {
        contact = false;
    }

}

double SourceRobotFrameMapper::outputHeight() const {
    return impl_->outputHeight;
}

struct OnlineCanonicalFitter::Impl {
    struct Record {
        std::size_t index = 0;
        HumanFrame frame;
        std::unordered_map<std::string, bool> rawContacts;
        std::unordered_map<std::string, bool> confirmedContacts;
    };

    CanonicalConfig config;
    double fps = 30.0;
    int latency = 2;
    std::size_t seen = 0;
    std::size_t nextOutput = 0;
    std::deque<Record> records;
    std::unordered_map<std::string, int> contactRunLengths;
    std::unordered_map<std::string, int> nonContactRunLengths;
    std::unordered_map<std::string, bool> stableContacts;
    std::unordered_map<std::string, Eigen::Quaterniond> previousOutputOrientations;
    double groundHeight = std::numeric_limits<double>::infinity();

    const Record& record(std::size_t index) const {
        return records.at(index - records.front().index);
    }

    Record& record(std::size_t index) {
        return records.at(index - records.front().index);
    }
};

OnlineCanonicalFitter::OnlineCanonicalFitter(
    const std::filesystem::path& mappingPath,
    double fps)
    : impl_(std::make_unique<Impl>()) {
    if (!std::filesystem::is_regular_file(mappingPath)) {
        throw std::runtime_error("Source mapping not found: " + mappingPath.string());
    }

    if (!std::isfinite(fps) || fps <= 0.0) {
        throw std::runtime_error("Online canonical fps must be positive and finite.");
    }

    impl_->config = loadCanonicalConfig(YAML::LoadFile(mappingPath.string()), fps);
    impl_->fps = fps;
    impl_->latency = std::max(
        impl_->config.smoothWindow / 2,
        impl_->config.minContactFrames - 1);
    reset();
}

OnlineCanonicalFitter::~OnlineCanonicalFitter() = default;

void OnlineCanonicalFitter::pushFrame(HumanFrame frame) {
    pushFrame(std::move(frame), {});
}

void OnlineCanonicalFitter::pushFrame(
    HumanFrame frame,
    const std::unordered_map<std::string, bool>& sourceFootContacts) {
    Impl::Record record;
    record.index = impl_->seen++;

    if (!impl_->records.empty()) {
        const HumanFrame& previous = impl_->records.back().frame;
        for (auto& [name, state] : frame) {
            const auto it = previous.find(name);
            if (it != previous.end() && it->second.orientation.coeffs().dot(state.orientation.coeffs()) < 0.0) {
                state.orientation.coeffs() *= -1.0;
            }

        }

    }

    double lower = std::numeric_limits<double>::infinity();
    for (const std::string& name : impl_->config.footNames) {
        lower = std::min(lower, frame.at(name).position.z());
    }

    impl_->groundHeight = std::min(impl_->groundHeight, lower);
    const double band = std::max(impl_->config.heightThreshold, 0.025);
    for (const std::string& name : impl_->config.footNames) {
        double speed = 0.0;
        if (!impl_->records.empty()) {
            speed = (frame.at(name).position - impl_->records.back().frame.at(name).position)
                .norm() * impl_->fps;
        }

        const bool active = sourceFootContacts.empty()
            ? frame.at(name).position.z() <= impl_->groundHeight + band &&
                speed < impl_->config.speedThreshold
            : sourceFootContacts.at(name);
        record.rawContacts[name] = active;
        record.confirmedContacts[name] = false;
    }

    impl_->records.push_back(std::move(record));
    impl_->records.back().frame = std::move(frame);

    for (const std::string& name : impl_->config.footNames) {
        const bool rawContact = impl_->records.back().rawContacts.at(name);
        if (rawContact) {
            impl_->nonContactRunLengths[name] = 0;
            const int runLength = ++impl_->contactRunLengths[name];
            if (impl_->stableContacts[name]) {
                impl_->records.back().confirmedContacts[name] = true;
            } else if (runLength >= impl_->config.minContactFrames) {
                impl_->stableContacts[name] = true;
                for (auto it = impl_->records.rbegin(); it != impl_->records.rend(); ++it) {
                    if (!it->rawContacts.at(name)) {
                        break;
                    }

                    it->confirmedContacts[name] = true;
                }
            }

            continue;
        }

        impl_->contactRunLengths[name] = 0;
        const int runLength = ++impl_->nonContactRunLengths[name];
        if (impl_->stableContacts[name]) {
            impl_->records.back().confirmedContacts[name] = true;
            if (runLength >= impl_->config.minContactFrames) {
                impl_->stableContacts[name] = false;
                for (auto it = impl_->records.rbegin(); it != impl_->records.rend(); ++it) {
                    if (it->rawContacts.at(name)) {
                        break;
                    }

                    it->confirmedContacts[name] = false;
                }
            }

        }

    }

}

bool OnlineCanonicalFitter::canPop(bool flush) const {
    if (impl_->nextOutput >= impl_->seen) {
        return false;
    }

    return flush || impl_->nextOutput + static_cast<std::size_t>(impl_->latency) < impl_->seen;
}

OnlineCanonicalOutput OnlineCanonicalFitter::popFrame(bool flush) {
    if (!canPop(flush)) {
        throw std::runtime_error("Online canonical output is not ready.");
    }

    const std::size_t index = impl_->nextOutput;
    const std::size_t half = static_cast<std::size_t>(impl_->config.smoothWindow / 2);
    const std::size_t begin = index > half ? index - half : 0;
    const std::size_t end = std::min(impl_->seen, index + half + 1);
    HumanFrame target;
    const HumanFrame& names = impl_->record(index).frame;
    for (const auto& [name, unused] : names) {
        HumanBodyState state;
        Eigen::Vector4d quaternion = Eigen::Vector4d::Zero();
        for (std::size_t t = begin; t < end; ++t) {
            const HumanBodyState& sample = impl_->record(t).frame.at(name);
            state.position += sample.position;
            quaternion += Eigen::Vector4d(
                sample.orientation.w(),
                sample.orientation.x(),
                sample.orientation.y(),
                sample.orientation.z());
        }

        state.position /= static_cast<double>(end - begin);
        quaternion /= static_cast<double>(end - begin);
        state.orientation = Eigen::Quaterniond(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3]).normalized();
        const auto previous = impl_->previousOutputOrientations.find(name);
        if (previous != impl_->previousOutputOrientations.end() &&
            previous->second.coeffs().dot(state.orientation.coeffs()) < 0.0) {
            state.orientation.coeffs() *= -1.0;
        }

        impl_->previousOutputOrientations[name] = state.orientation;
        target[name] = state;
    }

    const auto contacts = impl_->record(index).confirmedContacts;
    OnlineCanonicalOutput output;
    output.frame = fitCanonicalFrame(target, impl_->config);
    output.footContacts = contacts;
    ++impl_->nextOutput;

    const std::size_t keepFrom = impl_->nextOutput > half ? impl_->nextOutput - half : 0;
    while (!impl_->records.empty() && impl_->records.front().index < keepFrom) {
        impl_->records.pop_front();
    }

    return output;
}

void OnlineCanonicalFitter::reset() {
    impl_->seen = 0;
    impl_->nextOutput = 0;
    impl_->records.clear();
    impl_->contactRunLengths.clear();
    impl_->nonContactRunLengths.clear();
    impl_->stableContacts.clear();
    impl_->previousOutputOrientations.clear();
    impl_->groundHeight = std::numeric_limits<double>::infinity();
    for (const std::string& name : impl_->config.footNames) {
        impl_->contactRunLengths[name] = 0;
        impl_->nonContactRunLengths[name] = 0;
        impl_->stableContacts[name] = false;
    }
}

int OnlineCanonicalFitter::latencyFrames() const {
    return impl_->latency;
}

double OnlineCanonicalFitter::canonicalHeight() const {
    return impl_->config.height;
}

SourceRobotTrajectory loadSourceRobotTrajectory(
    const std::filesystem::path& inputPath,
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& gmrRoot,
    double fpsOverride,
    int maxFrames) {
    if (!std::filesystem::is_regular_file(inputPath)) {
        throw std::runtime_error("Source trajectory not found: " + inputPath.string());
    }

    if (!std::filesystem::is_regular_file(mappingPath)) {
        throw std::runtime_error("Source mapping not found: " + mappingPath.string());
    }

    const YAML::Node cfg = YAML::LoadFile(mappingPath.string());
    const std::filesystem::path modelPath = resolveConfigPath(cfg, "robot_model", gmrRoot);
    const ModelPtr model = loadModel(modelPath);

    std::vector<Eigen::VectorXd> frames;
    double inputFps = 0.0;
    if (inputPath.extension() == ".csv") {
        frames = parseCsvQpos(inputPath);
    } else if (inputPath.extension() == ".json") {
        frames = parseJsonQpos(inputPath, &inputFps);
    } else if (inputPath.extension() == ".npy") {
        frames = parseNpyQpos(inputPath);
    } else {
        throw std::runtime_error("Unsupported trajectory format: " + inputPath.extension().string());
    }

    if (maxFrames > 0 && frames.size() > static_cast<std::size_t>(maxFrames)) {
        frames.resize(maxFrames);
    }

    validateSourceFrames(frames, model->nq);
    const double fps = fpsOverride > 0.0
        ? fpsOverride
        : (inputFps > 0.0 ? inputFps : cfg["fps_default"].as<double>(30.0));
    if (!std::isfinite(fps) || fps <= 0.0) {
        throw std::runtime_error("fps must be positive and finite.");
    }

    return {std::move(frames), fps, inputPath.stem().string()};
}

CanonicalRobotTrajectory buildCanonicalRobotTrajectory(
    const SourceRobotTrajectory& source,
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& gmrRoot,
    bool groundAlign) {
    const YAML::Node yaml = YAML::LoadFile(mappingPath.string());
    const std::filesystem::path modelPath = resolveConfigPath(yaml, "robot_model", gmrRoot);
    const ModelPtr model = loadModel(modelPath);
    DataPtr data(mj_makeData(model.get()));
    if (!data) {
        throw std::runtime_error("Failed to allocate source MuJoCo data.");
    }

    const double canonicalHeight = yaml["canonical_height_m"].as<double>();
    const double sourceHeight = yaml["source_robot_reference_height_m"].as<double>();
    if (sourceHeight <= 0.0) {
        throw std::runtime_error("source_robot_reference_height_m must be positive.");
    }

    const double scale = canonicalHeight / sourceHeight;
    const bool scaleRootXy = yaml["scale_root_xy"].as<bool>(false);
    std::vector<SiteSpec> sites;
    const YAML::Node yamlSites = yaml["sites"];
    for (auto it = yamlSites.begin(); it != yamlSites.end(); ++it) {
        SiteSpec site;
        site.name = it->first.as<std::string>();
        const YAML::Node spec = it->second;
        const std::string bodyName = spec["source_body"].as<std::string>();
        site.bodyId = mj_name2id(model.get(), mjOBJ_BODY, bodyName.c_str());
        if (site.bodyId < 0) {
            throw std::runtime_error("Unknown source body '" + bodyName + "' for site '" + site.name + "'.");
        }

        site.positionOffset = yamlVec3(spec["position_offset_local"], site.name + ".position_offset_local");
        site.orientationOffset = yamlQuat(spec["orientation_offset_wxyz"], site.name + ".orientation_offset_wxyz");
        sites.push_back(std::move(site));
    }

    const std::array<std::string, 14> required = {
        "pelvis", "spine3", "left_hip", "right_hip", "left_knee", "right_knee", "left_foot",
        "right_foot", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"};
    for (const std::string& name : required) {
        if (std::none_of(sites.begin(), sites.end(), [&](const SiteSpec& site) { return site.name == name; })) {
            throw std::runtime_error("Source mapping is missing required semantic site: " + name);
        }
    }

    std::vector<HumanFrame> semanticFrames;
    semanticFrames.reserve(source.qposFrames.size());
    std::unordered_map<std::string, Eigen::Quaterniond> previousQuaternions;
    Eigen::Vector3d root0 = Eigen::Vector3d::Zero();
    for (std::size_t t = 0; t < source.qposFrames.size(); ++t) {
        mju_copy(data->qpos, source.qposFrames[t].data(), model->nq);
        mj_forward(model.get(), data.get());
        HumanFrame raw;
        for (const SiteSpec& site : sites) {
            const Eigen::Quaterniond linkOrientation = bodyQuaternion(data.get(), site.bodyId);
            HumanBodyState state;
            state.position = bodyPosition(data.get(), site.bodyId) + linkOrientation * site.positionOffset;
            state.orientation = (linkOrientation * site.orientationOffset).normalized();
            raw[site.name] = state;
        }

        const Eigen::Vector3d pelvis = raw.at("pelvis").position;
        if (t == 0) {
            root0 = pelvis;
        }

        Eigen::Vector3d scaledPelvis = pelvis;
        scaledPelvis.z() = root0.z() * scale + (pelvis.z() - root0.z()) * scale;
        if (scaleRootXy) {
            scaledPelvis.head<2>() = root0.head<2>() + (pelvis - root0).head<2>() * scale;
        }

        HumanFrame scaled;
        for (auto& [name, state] : raw) {
            state.position = scaledPelvis + (state.position - pelvis) * scale;
            auto previous = previousQuaternions.find(name);
            if (previous != previousQuaternions.end() && previous->second.coeffs().dot(state.orientation.coeffs()) < 0.0) {
                state.orientation.coeffs() *= -1.0;
            }

            previousQuaternions[name] = state.orientation;
            scaled[name] = state;
        }

        semanticFrames.push_back(std::move(scaled));
    }

    const CanonicalConfig cfg = loadCanonicalConfig(yaml, source.fps);
    const Contacts contacts = inferSourceRobotContacts(
        model.get(),
        data.get(),
        source.qposFrames,
        cfg);
    std::vector<HumanFrame> targets = smoothFrames(semanticFrames, cfg.smoothWindow);

    std::vector<HumanFrame> fitted;
    fitted.reserve(targets.size());
    for (const HumanFrame& frame : targets) {
        fitted.push_back(fitCanonicalFrame(frame, cfg));
    }

    const CanonicalFitQuality quality = measureQuality(semanticFrames, fitted, contacts, source.fps);
    if (groundAlign) {
        groundAlignFrames(fitted);
    }

    CanonicalRobotTrajectory out;
    out.sequence.frames = std::move(fitted);
    out.sequence.footContacts = contacts;
    out.sequence.fps = static_cast<int>(std::lround(source.fps));
    out.sequence.srcHuman = "smplx";
    out.sequence.actualHumanHeight = canonicalHeight;
    out.quality = quality;
    out.canonicalHeight = canonicalHeight;
    out.globalScale = scale;
    return out;
}

RobotPostprocessResult postprocessRobotTrajectory(
    std::vector<Eigen::VectorXd>& qposFrames,
    const std::filesystem::path& robotModelPath,
    const Contacts& contacts,
    double fps,
    bool alignWristsEnabled) {
    const ModelPtr model = loadModel(robotModelPath);
    DataPtr data(mj_makeData(model.get()));
    if (!data) {
        throw std::runtime_error("Failed to allocate target MuJoCo data.");
    }

    if (qposFrames.empty() || qposFrames.size() != contacts.size()) {
        throw std::runtime_error("Postprocess requires one contact entry per non-empty qpos frame.");
    }

    for (const Eigen::VectorXd& q : qposFrames) {
        if (q.size() != model->nq) {
            throw std::runtime_error("Target qpos size does not match target MuJoCo model.nq.");
        }
    }

    const auto feet = resolveFootReferences(model.get());
    RobotPostprocessResult result;
    result.stanceSlipBeforeMps = measureStanceSlip(
        model.get(), data.get(), qposFrames, contacts, fps, feet);
    plantStanceFeet(model.get(), data.get(), qposFrames, contacts, feet);
    flattenStanceFeet(model.get(), data.get(), qposFrames, contacts, feet, fps);
    snapToGround(model.get(), data.get(), qposFrames, contacts, feet);
    if (alignWristsEnabled) {
        alignWrists(model.get(), data.get(), qposFrames);
    }

    result.stanceSlipAfterMps = measureStanceSlip(
        model.get(), data.get(), qposFrames, contacts, fps, feet);
    return result;
}

void writeCanonicalRobotTrajectory(
    const std::filesystem::path& outputPath,
    const CanonicalRobotTrajectory& trajectory) {
    nlohmann::json root;
    root["schema_version"] = "gmr_reference_v1";
    root["fps"] = trajectory.sequence.fps;
    root["src_human"] = trajectory.sequence.srcHuman;
    root["actual_human_height"] = trajectory.sequence.actualHumanHeight;
    root["contacts"] = trajectory.sequence.footContacts;
    root["quality"] = {
        {"semantic_position_rmse_m", trajectory.quality.semanticPositionRmseM},
        {"semantic_position_p95_m", trajectory.quality.semanticPositionP95M},
        {"semantic_rotation_mean_deg", trajectory.quality.semanticRotationMeanDeg},
        {"contact_slip_mean_mps", trajectory.quality.contactSlipMeanMps},
    };

    root["frames"] = nlohmann::json::array();
    for (const HumanFrame& frame : trajectory.sequence.frames) {
        nlohmann::json frameJson;
        for (const auto& [name, state] : frame) {
            frameJson[name] = bodyStateJson(state);
        }

        root["frames"].push_back(std::move(frameJson));
    }

    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    std::ofstream output(outputPath);
    if (!output) {
        throw std::runtime_error("Failed to open canonical output: " + outputPath.string());
    }

    output << root.dump(2) << '\n';
}

}  // namespace gmr
