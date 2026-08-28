#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "gmr/retarget/human_frame_io.h"

namespace gmr {

struct SourceRobotTrajectory {
    std::vector<Eigen::VectorXd> qposFrames;
    double fps = 30.0;
    std::string sourceId;
};

struct CanonicalFitQuality {
    double semanticPositionRmseM = 0.0;
    double semanticPositionP95M = 0.0;
    double semanticRotationMeanDeg = 0.0;
    double contactSlipMeanMps = 0.0;
};

struct CanonicalRobotTrajectory {
    HumanFrameSequence sequence;
    CanonicalFitQuality quality;
    double canonicalHeight = 1.8;
    double globalScale = 1.0;
};

SourceRobotTrajectory loadSourceRobotTrajectory(
    const std::filesystem::path& inputPath,
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& gmrRoot,
    double fpsOverride = 0.0,
    int maxFrames = 0);

CanonicalRobotTrajectory buildCanonicalRobotTrajectory(
    const SourceRobotTrajectory& source,
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& gmrRoot,
    bool groundAlign = true);

struct RobotPostprocessResult {
    double stanceSlipBeforeMps = 0.0;
    double stanceSlipAfterMps = 0.0;
};

RobotPostprocessResult postprocessRobotTrajectory(
    std::vector<Eigen::VectorXd>& qposFrames,
    const std::filesystem::path& robotModelPath,
    const std::vector<std::unordered_map<std::string, bool>>& contacts,
    double fps,
    bool alignWrists = true);

void writeCanonicalRobotTrajectory(
    const std::filesystem::path& outputPath,
    const CanonicalRobotTrajectory& trajectory);

}  // namespace gmr
