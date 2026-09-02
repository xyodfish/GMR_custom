#pragma once

#include <filesystem>
#include <memory>
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

/// Stateful source-robot FK mapper for causal robot-to-robot retargeting.
/// It emits scaled semantic sites directly and deliberately skips canonical fitting,
/// trajectory smoothing, contact lookahead, and ground postprocessing.
class SourceRobotFrameMapper {
   public:
    SourceRobotFrameMapper(
        const std::filesystem::path& mappingPath,
        const std::filesystem::path& gmrRoot);
    ~SourceRobotFrameMapper();

    SourceRobotFrameMapper(const SourceRobotFrameMapper&) = delete;
    SourceRobotFrameMapper& operator=(const SourceRobotFrameMapper&) = delete;

    HumanFrame mapFrame(const Eigen::VectorXd& qpos);
    const std::unordered_map<std::string, bool>& footContacts() const;
    void reset();

    double outputHeight() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct OnlineCanonicalOutput {
    HumanFrame frame;
    std::unordered_map<std::string, bool> footContacts;
};

/// Fixed-latency streaming counterpart of buildCanonicalRobotTrajectory().
/// It preserves canonical torso/leg lengths and centered semantic smoothing, and emits confirmed contacts.
class OnlineCanonicalFitter {
   public:
    OnlineCanonicalFitter(const std::filesystem::path& mappingPath, double fps);
    ~OnlineCanonicalFitter();

    OnlineCanonicalFitter(const OnlineCanonicalFitter&) = delete;
    OnlineCanonicalFitter& operator=(const OnlineCanonicalFitter&) = delete;

    void pushFrame(HumanFrame frame);
    void pushFrame(
        HumanFrame frame,
        const std::unordered_map<std::string, bool>& sourceFootContacts);
    bool canPop(bool flush = false) const;
    OnlineCanonicalOutput popFrame(bool flush = false);
    void reset();

    int latencyFrames() const;
    double canonicalHeight() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

SourceRobotTrajectory loadSourceRobotTrajectory(
    const std::filesystem::path& inputPath,
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& gmrRoot,
    double fpsOverride = 0.0,
    int maxFrames = 0);

/// Stateful one-frame mapper for robots with identical named one-DoF joints.
class CompatibleRobotMapper {
   public:
    CompatibleRobotMapper(
        const std::filesystem::path& sourceModelPath,
        const std::filesystem::path& targetModelPath);
    ~CompatibleRobotMapper();

    CompatibleRobotMapper(const CompatibleRobotMapper&) = delete;
    CompatibleRobotMapper& operator=(const CompatibleRobotMapper&) = delete;

    Eigen::VectorXd mapFrame(const Eigen::VectorXd& sourceQpos);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Copy a trajectory with CompatibleRobotMapper.
std::vector<Eigen::VectorXd> mapCompatibleRobotTrajectory(
    const std::vector<Eigen::VectorXd>& sourceQpos,
    const std::filesystem::path& sourceModelPath,
    const std::filesystem::path& targetModelPath);

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
