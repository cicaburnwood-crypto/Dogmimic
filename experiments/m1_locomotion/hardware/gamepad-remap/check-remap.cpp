#include "robot/ControlPath/include/HighLevelPath.h"
#include "robot/hardwares/sbus/include/G20Types.h"
#include "robot_ctrl/controller/include/StandUpController.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
FsmState selected;
std::array<float, 3> received;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void near(double got, double expected) {
    require(std::abs(got - expected) < 1e-6, "Mapped axis differs from expected value");
}

HardwareState settledStand() {
    HardwareState state{};
    state.imu_valid = state.imu_estimate.valid = true;
    state.imu_estimate.q_ori_ << 1., 0., 0., 0.;
    state.imu_estimate.r_b_.setIdentity();
    state.imu_estimate.omega_b_.setZero();
    state.imu_estimate.a_w_.setZero();
    for (std::size_t leg = 0; leg < 4; ++leg) {
        auto pose = ControllerConfig::standPose(leg);
        const float side = leg < 2 ? 1.F : -1.F;
        pose[1] -= .026F * side;
        pose[2] -= .124F * side;
        for (std::size_t axis = 0; axis < 4; ++axis) {
            auto& joint = state.joints[leg * 4 + axis];
            joint.valid = true;
            joint.position_rad = axis < 3 ? pose[axis] : 0.F;
        }
    }
    return state;
}
}

extern "C" JointCmd __wrap__ZN10Controller7computeERK13HardwareState8FsmStateRK11PathCommandb(
    Controller*, const HardwareState&, FsmState mode, const PathCommand& command, bool) {
    selected = mode;
    received = command.high().v_des;
    return {};
}

extern "C" void __wrap__ZN16StateVoicePlayer7requestENS_3CueE(StateVoicePlayer*, StateVoicePlayer::Cue) noexcept {}

int main() {
    try {
        using namespace g20_controller;
        const auto state = settledStand();
        require(StandUpController::isSettled(state), "Test posture is not standard standing");
        HighLevelPath path;
        path.onEnter(state);
        PathCommand command;
        command.valid = true;
        command.remote_disconnected = false;
        auto& requested = command.high();

        const auto send = [&](CommandMode mode, FsmState expected, std::array<float, 3> input,
                              bool l2 = false, unsigned device = GAMEPAD) {
            requested.requested_mode = mode;
            requested.control_device = device;
            requested.uni_exp_start = l2;
            requested.v_des = input;
            (void)path.step(state, command);
            require(selected == expected, "Unexpected active mode");
            for (std::size_t i = 0; i < input.size(); ++i) {
                require(requested.v_des[i] == input[i] ||
                        (std::isnan(requested.v_des[i]) && std::isnan(input[i])), "Original input was mutated");
            }
        };

        const std::array<float, 3> original{.76F, .65F, -.81F};
        send(CommandMode::LocoUniExp, FsmState::Passive, original, true);
        require(received == original, "Passive command was remapped");
        for (int i = 0; i < 1600; ++i) send(CommandMode::StandUp, FsmState::StandUp, {});
        send(CommandMode::LocoUniExp, FsmState::StandUp, {0.F, .08F, 0.F}, true);
        near(received[1], .08F);
        send(CommandMode::LocoUniExp, FsmState::LocoUniExp, {}, true);
        require(received == std::array<float, 3>{}, "Neutral handoff changed");

        for (float sign : {-1.F, 1.F}) {
            send(CommandMode::LocoUniExp, FsmState::LocoUniExp, {sign * 1.5F, 0.F, 0.F});
            near(1.4 * received[0], sign * 1.4);
            send(CommandMode::LocoUniExp, FsmState::LocoUniExp, {0.F, sign * 1.5F, 0.F});
            near(.5 * received[1], sign * .22);
            send(CommandMode::LocoUniExp, FsmState::LocoUniExp, {0.F, 0.F, sign * 1.5F});
            near(.8 * received[2], sign * .35);
        }
        send(CommandMode::LocoUniExp, FsmState::LocoUniExp, {1.F, 0.F, 1.F});
        near(1.4 * received[0], .70);
        near(.8 * received[2], .175);

        int cases = 0;
        const std::array<float, 7> grid{-1.5F, -1.F, -.5F, 0.F, .5F, 1.F, 1.5F};
        for (float x : grid) for (float y : grid) for (float yaw : grid) {
            send(CommandMode::LocoUniExp, FsmState::LocoUniExp, {x, y, yaw});
            require(std::abs(received[0]) + std::abs(.5 * received[1] / .22) +
                    std::abs(.8 * received[2] / .35) <= 1.000001, "Combined training budget exceeded");
            ++cases;
        }
        send(CommandMode::LocoUniExp, FsmState::LocoUniExp,
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -.5F});
        near(received[0], 0.);
        near(received[1], 0.);
        near(.8 * received[2], -.175);

        send(CommandMode::LocoUniExp, FsmState::LocoUniExp, original, false, HIGHLEVELSDK);
        require(received == original, "SDK input was remapped");
        send(CommandMode::LocoClassic, FsmState::LocoClassic, original);
        require(received == original, "Classic input was remapped");
        send(CommandMode::LocoUniExp, FsmState::LocoClassic, original, true);
        require(received == original, "Rejected L2 request remapped Classic input");
        send(CommandMode::LocoStairs, FsmState::LocoStairs, original);
        require(received == original, "Stairs input was remapped");
        send(CommandMode::Damping, FsmState::Damping, original);
        require(received == original, "Damping input was remapped");
        path.onExit();
        std::cout << "GAMEPAD_REMAP_OK grid " << cases << " full_axes 1.4 0.22 0.35"
                  << " full_forward_yaw 0.70 0.175 unchanged Classic Stairs Damping Passive SDK\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
