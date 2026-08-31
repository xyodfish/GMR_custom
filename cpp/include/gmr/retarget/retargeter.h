#pragma once

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "gmr/retarget/human_frame_types.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/contact_ground.h"

namespace gmr {

    struct RetargetOptions {
        std::string solverName     = "daqp";
        double damping             = 5e-1;
        /// Integration dt for IK; <= 0 uses MuJoCo ``model.opt.timestep`` (Python ``mink.solve_ik``).
        double integrationTimestep = 0.0;
        int maxIterations          = 15;
        /// Per-task LM scale (mink ``FrameTask`` ``lm_damping``); mu = scale * ||W e||^2 per task.
        double taskLmDamping = 1.0;
        /// Mink ``ConfigurationLimit`` gain for joint position bounds.
        double configurationLimitGain = 0.95;
        bool useVelocityLimit      = false;
        double velocityLimit       = 3.0 * M_PI;
        double progressThreshold   = 1e-3;
        double motionFps           = 30.0;
        ContactGroundConfig contactGround;
    };

    struct ScalarJointCoordinate {
        int qIndex = -1;
        int vIndex = -1;
        std::string jointName;
    };

    enum class RetargetBackend {
        kPinocchio,
        kMujoco,
    };

    RetargetBackend parseRetargetBackend(const std::string& backendName);
    const char* toString(RetargetBackend backend);

    class Retargeter {
       public:
        virtual ~Retargeter() = default;

        virtual Eigen::VectorXd retargetFrame(const HumanFrame& humanFrame, bool offsetToGround = false)      = 0;
        virtual HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround = false) const = 0;
        /// Scale/offset human frame plus contact-ground preprocessing (same as ``retargetFrame`` input).
        virtual HumanFrame prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround = false)    = 0;
        /// Same preparation with externally established contact labels.
        virtual HumanFrame prepareRetargetInput(
            const HumanFrame& humanFrame,
            const ContactGroundState& contactState,
            bool offsetToGround = false) = 0;
        virtual Eigen::VectorXd retargetPreparedFrame(const HumanFrame& rawFrame, const HumanFrame& preparedFrame) = 0;
        virtual Eigen::VectorXd retargetPreparedLightIk(const HumanFrame& rawFrame, const HumanFrame& preparedFrame,
                                                        int maxIterations) = 0;
        /// Few IK iterations from current ``qpos`` (Python ``_light_ik_warmstart``).
        virtual Eigen::VectorXd retargetLightIk(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) = 0;
        virtual void setQpos(const Eigen::VectorXd& qpos)                                                     = 0;
        /// Apply contact-ground penetration fix on current qpos (no-op if disabled).
        virtual void finalizeContact()                                                                          = 0;
        virtual void finalizeContact(const ContactGroundState& state)                                           = 0;
        virtual ContactGroundState contactGroundState() const                                                   = 0;

        virtual const Eigen::VectorXd& currentQpos() const                               = 0;
        virtual bool hasRootFreeFlyer() const                                            = 0;
        virtual const std::vector<ScalarJointCoordinate>& scalarJointCoordinates() const = 0;
        virtual void setMotionFps(double fps)                                            = 0;
    };

    class PinocchioRetargetBackend final : public Retargeter {
       public:
        PinocchioRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig, RetargetOptions options = {});
        ~PinocchioRetargetBackend() override;

        PinocchioRetargetBackend(const PinocchioRetargetBackend&) = delete;
        PinocchioRetargetBackend& operator=(const PinocchioRetargetBackend&) = delete;

        Eigen::VectorXd retargetFrame(const HumanFrame& humanFrame, bool offsetToGround = false) override;
        HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround = false) const override;
        HumanFrame prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround = false) override;
        HumanFrame prepareRetargetInput(
            const HumanFrame& humanFrame,
            const ContactGroundState& contactState,
            bool offsetToGround = false) override;
        Eigen::VectorXd retargetPreparedFrame(const HumanFrame& rawFrame, const HumanFrame& preparedFrame) override;
        Eigen::VectorXd retargetPreparedLightIk(const HumanFrame& rawFrame, const HumanFrame& preparedFrame,
                                                int maxIterations) override;
        Eigen::VectorXd retargetLightIk(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) override;
        void setQpos(const Eigen::VectorXd& qpos) override;
        void finalizeContact() override;
        void finalizeContact(const ContactGroundState& state) override;
        ContactGroundState contactGroundState() const override;

        const Eigen::VectorXd& currentQpos() const override;
        bool hasRootFreeFlyer() const override;
        const std::vector<ScalarJointCoordinate>& scalarJointCoordinates() const override;
        void setMotionFps(double fps) override;

       private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class MujocoRetargetBackend final : public Retargeter {
       public:
        MujocoRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig, RetargetOptions options = {});
        ~MujocoRetargetBackend() override;

        MujocoRetargetBackend(const MujocoRetargetBackend&) = delete;
        MujocoRetargetBackend& operator=(const MujocoRetargetBackend&) = delete;

        Eigen::VectorXd retargetFrame(const HumanFrame& humanFrame, bool offsetToGround = false) override;
        HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround = false) const override;
        HumanFrame prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround = false) override;
        HumanFrame prepareRetargetInput(
            const HumanFrame& humanFrame,
            const ContactGroundState& contactState,
            bool offsetToGround = false) override;
        Eigen::VectorXd retargetPreparedFrame(const HumanFrame& rawFrame, const HumanFrame& preparedFrame) override;
        Eigen::VectorXd retargetPreparedLightIk(const HumanFrame& rawFrame, const HumanFrame& preparedFrame,
                                                int maxIterations) override;
        Eigen::VectorXd retargetLightIk(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) override;
        void setQpos(const Eigen::VectorXd& qpos) override;
        void finalizeContact() override;
        void finalizeContact(const ContactGroundState& state) override;
        ContactGroundState contactGroundState() const override;

        const Eigen::VectorXd& currentQpos() const override;
        bool hasRootFreeFlyer() const override;
        const std::vector<ScalarJointCoordinate>& scalarJointCoordinates() const override;
        void setMotionFps(double fps) override;

       private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::unique_ptr<Retargeter> createRetargeter(RetargetBackend backend, const std::filesystem::path& robotModelPath, IkConfig ikConfig,
                                                 RetargetOptions options = {});

}  // namespace gmr
