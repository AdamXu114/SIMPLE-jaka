#include "RLController.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace RLController
{

    extern fsm_func_ptr StateJudge;
    extern fsm_func_ptr StateRun;
    extern FSMSTATE ReqState;

    namespace
    {
        constexpr int kJakaMiniZMQCommandDim = 155;
        constexpr int kJakaMiniZMQAnchorOriDim = 30;
        constexpr int kJakaMiniZMQGravityDim = 3;
        constexpr int kJakaMiniZMQAngVelDim = 3;
        constexpr int kJakaMiniZMQHistObsDim = 87;
        constexpr int kJakaMiniZMQFutureSteps = 5;
        constexpr int kJakaMiniZMQRot6dStride = 6;
        constexpr int kJakaMiniZMQDefaultFrameStack = 5;
        constexpr float kJakaMiniZMQPi = 3.14159265358979323846f;
        constexpr std::array<int, 27> kJakaMiniZMQIsaaclabToDeployIndex = {
            0, 12, 24,
            1, 13, 6, 25, 18,
            2, 14, 7, 26, 19,
            3, 15, 8, 20,
            4, 16, 9, 21,
            5, 17, 10, 22,
            11, 23};

    } // anonymous namespace

    static legged_waist_controller::RobotIntf::JoyRobotState zmq_joydata;
    static legged_waist_controller::RobotIntf::ImuData zmq_imu;
    static double zmq_q_cmd[50], zmq_q[50], zmq_qd[50];
    static double zmq_omega[3], zmq_acc[3], zmq_quat[4];
    static double zmq_pitch, zmq_roll, zmq_dpitch, zmq_droll, zmq_J[2][2];
    static bool zmq_play;
    static bool zmq_first_step;
    static bool zmq_last_ready;
    static int zmq_frame_stack;
    static Eigen::Quaternionf zmq_align_quat(1.0f, 0.0f, 0.0f, 0.0f);
    static std::vector<float> zmq_single_obs;
    static std::vector<float> zmq_stacked_obs;
    static std::vector<float> zmq_last_action;
    static std::deque<std::vector<float>> zmq_obs_history;

    static std::unique_ptr<RealtimeMotionBuffer> zmq_motion_buffer;

    static void from_fsm_mimic_jaka_mini_zmq();

    static inline Eigen::Quaternionf zmq_normalize_quat(const Eigen::Quaternionf &q)
    {
        if (q.norm() < 1.0e-6f)
            return Eigen::Quaternionf::Identity();
        Eigen::Quaternionf r = q;
        r.normalize();
        return r;
    }

    static inline Eigen::Vector3f zmq_quat_rotate_inverse(const Eigen::Quaternionf &q, const Eigen::Vector3f &v)
    {
        return zmq_normalize_quat(q).conjugate() * v;
    }

    static inline std::array<float, kJakaMiniZMQRot6dStride> zmq_quat_to_rot6d(const Eigen::Quaternionf &q)
    {
        const Eigen::Matrix3f rot = zmq_normalize_quat(q).toRotationMatrix();
        return {rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1)};
    }

    static inline Eigen::Quaternionf zmq_robot_quat_from_imu()
    {
        return zmq_normalize_quat(Eigen::Quaternionf(
            static_cast<float>(zmq_quat[0]), static_cast<float>(zmq_quat[1]),
            static_cast<float>(zmq_quat[2]), static_cast<float>(zmq_quat[3])));
    }

    static inline bool zmq_resolve_policy_layout()
    {
        if (mMimicJakaMiniZmqPolicy.session == nullptr)
        {
            std::cerr << "[FSMJakaMiniZmq] policy session is null." << std::endl;
            return false;
        }

        zmq_frame_stack = kJakaMiniZMQDefaultFrameStack;
        if (mMimicJakaMiniZmqPolicy.input_shape[1] > 0)
        {
            const int non_history_dim = kJakaMiniZMQCommandDim + kJakaMiniZMQAnchorOriDim;
            const int remaining = static_cast<int>(mMimicJakaMiniZmqPolicy.input_shape[1]) - non_history_dim;
            if (remaining <= 0 || remaining % kJakaMiniZMQHistObsDim != 0)
            {
                std::cerr << "[FSMJakaMiniZmq] input dim mismatch" << std::endl;
                return false;
            }
            zmq_frame_stack = remaining / kJakaMiniZMQHistObsDim;
        }

        if (static_cast<int>(mMimicJakaMiniZmqPolicy.output_shape[1]) != mNumAction)
        {
            std::cerr << "[FSMJakaMiniZmq] output dim mismatch" << std::endl;
            return false;
        }

        zmq_single_obs.assign(kJakaMiniZMQHistObsDim, 0.0f);
        zmq_stacked_obs.assign(static_cast<int>(mMimicJakaMiniZmqPolicy.input_shape[1]), 0.0f);
        zmq_last_action.assign(mNumAction, 0.0f);
        zmq_obs_history.clear();

        std::cerr << "[FSMJakaMiniZmq] action_dim=" << mNumAction
                  << ", frame_stack=" << zmq_frame_stack << std::endl;
        return true;
    }

    static inline bool zmq_fetch_robot_state()
    {
        update_joint_status(zmq_q_cmd, zmq_q, zmq_qd);
        if (mEnabledJointCount != mNumAction || mFaultJointCount > 0)
            return false;

        if (AB2PitchRoll(zmq_q[4], zmq_q[5], zmq_qd[4], zmq_qd[5], &zmq_pitch, &zmq_roll, &zmq_dpitch, &zmq_droll, zmq_J) < 0)
        {
            zmq_pitch = 0.0;
            zmq_roll = 0.0;
            zmq_dpitch = 0.0;
            zmq_droll = 0.0;
        }
        zmq_q[4] = zmq_pitch;
        zmq_q[5] = zmq_roll;
        zmq_qd[4] = zmq_dpitch;
        zmq_qd[5] = zmq_droll;

        if (AB2PitchRoll(-zmq_q[16], -zmq_q[17], -zmq_qd[16], -zmq_qd[17], &zmq_pitch, &zmq_roll, &zmq_dpitch, &zmq_droll, zmq_J) < 0)
        {
            zmq_pitch = 0.0;
            zmq_roll = 0.0;
            zmq_dpitch = 0.0;
            zmq_droll = 0.0;
        }
        zmq_q[16] = zmq_pitch;
        zmq_q[17] = zmq_roll;
        zmq_qd[16] = zmq_dpitch;
        zmq_qd[17] = zmq_droll;

        if (mRobot->get_imu(zmq_imu))
        {
            zmq_acc[0] = zmq_imu.ax;
            zmq_acc[1] = zmq_imu.ay;
            zmq_acc[2] = zmq_imu.az;
            zmq_omega[0] = zmq_imu.wx;
            zmq_omega[1] = zmq_imu.wy;
            zmq_omega[2] = zmq_imu.wz;
            zmq_quat[0] = zmq_imu.qw;
            zmq_quat[1] = zmq_imu.qx;
            zmq_quat[2] = zmq_imu.qy;
            zmq_quat[3] = zmq_imu.qz;
            imu_process(zmq_acc, zmq_omega, zmq_quat);
        }
        else
        {
            zmq_acc[0] = zmq_acc[1] = zmq_acc[2] = 0.0;
            zmq_omega[0] = zmq_omega[1] = zmq_omega[2] = 0.0;
            zmq_quat[0] = 1.0;
            zmq_quat[1] = zmq_quat[2] = zmq_quat[3] = 0.0;
        }
        return true;
    }

    static inline void zmq_initialize_alignment(const InterpolatedFrame &frame)
    {
        const Eigen::Quaternionf rob_yaw0 = yawQuaternion(zmq_robot_quat_from_imu());
        const Eigen::Quaternionf ref_yaw0 = yawQuaternion(zmq_normalize_quat(frame.root_quat_w));
        zmq_align_quat = zmq_normalize_quat(rob_yaw0 * ref_yaw0.conjugate());
    }

    static inline void zmq_initialize_last_action()
    {
        for (int isaac_idx = 0; isaac_idx < mNumAction; ++isaac_idx)
        {
            const int deploy_idx = kJakaMiniZMQIsaaclabToDeployIndex[isaac_idx];
            zmq_last_action[isaac_idx] =
                static_cast<float>((zmq_q_cmd[deploy_idx] - mDefaultPos[deploy_idx]) / mActionScale);
        }
    }

    static inline std::array<float, kJakaMiniZMQCommandDim> zmq_construct_command(
        const std::array<InterpolatedFrame, RealtimeMotionBuffer::kFutureSteps> &frames)
    {
        std::array<float, kJakaMiniZMQCommandDim> cmd{};
        const Eigen::Vector3f anchor_pos = frames[0].root_pos_w;
        const Eigen::Quaternionf anchor_quat = zmq_normalize_quat(frames[0].root_quat_w);

        int offset = 0;
        for (int i = 0; i < kJakaMiniZMQFutureSteps; ++i)
        {
            const Eigen::Vector3f diff_w = frames[i].root_pos_w - anchor_pos;
            const Eigen::Vector3f diff_b = zmq_quat_rotate_inverse(anchor_quat, diff_w);
            cmd[offset++] = diff_b[0];
            cmd[offset++] = diff_b[1];
            cmd[offset++] = diff_b[2];
        }

        for (int i = 0; i < kJakaMiniZMQFutureSteps; ++i)
            cmd[offset++] = frames[i].root_pos_w[2];

        for (int i = 0; i < kJakaMiniZMQFutureSteps; ++i)
        {
            for (int isaac_idx = 0; isaac_idx < mNumAction; ++isaac_idx)
            {
                float jv = 0.0f;
                if (isaac_idx < frames[i].dof_pos.size())
                    jv = frames[i].dof_pos[isaac_idx];
                cmd[offset++] = jv;
            }
        }

        return cmd;
    }

    static inline std::array<float, kJakaMiniZMQAnchorOriDim> zmq_construct_anchor_orientation(
        const std::array<InterpolatedFrame, RealtimeMotionBuffer::kFutureSteps> &frames)
    {
        const Eigen::Quaternionf robot_quat_now = zmq_robot_quat_from_imu();
        std::array<float, kJakaMiniZMQAnchorOriDim> ori{};

        int offset = 0;
        for (int i = 0; i < kJakaMiniZMQFutureSteps; ++i)
        {
            const Eigen::Quaternionf ref_quat = zmq_normalize_quat(frames[i].root_quat_w);
            const Eigen::Quaternionf anchor_world = zmq_normalize_quat(zmq_align_quat * ref_quat);
            const Eigen::Quaternionf ori_body = zmq_normalize_quat(robot_quat_now.conjugate() * anchor_world);
            const auto rot6d = zmq_quat_to_rot6d(ori_body);
            for (int j = 0; j < kJakaMiniZMQRot6dStride; ++j)
                ori[offset++] = rot6d[j];
        }
        return ori;
    }

    static inline std::array<float, kJakaMiniZMQGravityDim> zmq_construct_gravity()
    {
        return {
            static_cast<float>(2.0 * (-zmq_quat[3] * zmq_quat[1] + zmq_quat[0] * zmq_quat[2])),
            static_cast<float>(-2.0 * (zmq_quat[3] * zmq_quat[2] + zmq_quat[0] * zmq_quat[1])),
            static_cast<float>(1.0 - 2.0 * (zmq_quat[0] * zmq_quat[0] + zmq_quat[3] * zmq_quat[3])),
        };
    }

    static inline void zmq_construct_single_obs()
    {
        const auto gravity = zmq_construct_gravity();
        std::fill(zmq_single_obs.begin(), zmq_single_obs.end(), 0.0f);

        int off = 0;
        std::copy(gravity.begin(), gravity.end(), zmq_single_obs.begin() + off);
        off += kJakaMiniZMQGravityDim;

        for (int i = 0; i < kJakaMiniZMQAngVelDim; ++i)
            zmq_single_obs[off + i] = static_cast<float>(zmq_omega[i] * 0.25);
        off += kJakaMiniZMQAngVelDim;

        for (int isaac_idx = 0; isaac_idx < mNumAction; ++isaac_idx)
        {
            const int deploy_idx = kJakaMiniZMQIsaaclabToDeployIndex[isaac_idx];
            zmq_single_obs[off + isaac_idx] = static_cast<float>(zmq_q[deploy_idx] - mDefaultPos[deploy_idx]);
        }
        off += mNumAction;

        for (int isaac_idx = 0; isaac_idx < mNumAction; ++isaac_idx)
        {
            const int deploy_idx = kJakaMiniZMQIsaaclabToDeployIndex[isaac_idx];
            zmq_single_obs[off + isaac_idx] = static_cast<float>(zmq_qd[deploy_idx] * 0.05);
        }
        off += mNumAction;

        for (int i = 0; i < mNumAction; ++i)
            zmq_single_obs[off + i] = zmq_last_action[i];
    }

    static inline void zmq_init_history()
    {
        zmq_obs_history.clear();
        for (int i = 0; i < zmq_frame_stack; ++i)
            zmq_obs_history.push_back(zmq_single_obs);
    }

    static inline void zmq_push_history()
    {
        if (static_cast<int>(zmq_obs_history.size()) == zmq_frame_stack)
            zmq_obs_history.pop_front();
        zmq_obs_history.push_back(zmq_single_obs);
    }

    static inline void zmq_pack_history()
    {
        auto frames = zmq_motion_buffer->get_obs();
        const auto command = zmq_construct_command(frames);
        const auto anchor_ori = zmq_construct_anchor_orientation(frames);

        static constexpr std::array<int, 5> hist_blocks = {3, 3, 27, 27, 27};

        std::fill(zmq_stacked_obs.begin(), zmq_stacked_obs.end(), 0.0f);

        int w = 0;
        std::copy(command.begin(), command.end(), zmq_stacked_obs.begin() + w);
        w += kJakaMiniZMQCommandDim;
        std::copy(anchor_ori.begin(), anchor_ori.end(), zmq_stacked_obs.begin() + w);
        w += kJakaMiniZMQAnchorOriDim;

        int r = 0;
        for (int bs : hist_blocks)
        {
            for (const auto &frame : zmq_obs_history)
            {
                std::copy_n(frame.begin() + r, bs, zmq_stacked_obs.begin() + w);
                w += bs;
            }
            r += bs;
        }

        std::copy(zmq_stacked_obs.begin(), zmq_stacked_obs.end(),
                  mMimicJakaMiniZmqPolicy.input_buffer);
    }

    static inline double zmq_clamp_target(double target, int idx)
    {
        return std::max(mDownLmt[idx], std::min(mUpLmt[idx], target));
    }

    static inline void zmq_fill_trajectory()
    {
        for (int isaac_idx = 0; isaac_idx < mNumAction; ++isaac_idx)
        {
            const int deploy_idx = kJakaMiniZMQIsaaclabToDeployIndex[isaac_idx];
            zmq_last_action[isaac_idx] =
                std::clamp(mMimicJakaMiniZmqPolicy.output_buffer[isaac_idx], -5.0f, 5.0f);
            mPt.positions[deploy_idx] =
                static_cast<double>(zmq_last_action[isaac_idx]) * mActionScale + mDefaultPos[deploy_idx];
        }

        if (mAngleMapFlag)
        {
            double a, b;
            PitchRoll2AB(mPt.positions[4], mPt.positions[5], &a, &b);
            mPt.positions[4] = a;
            mPt.positions[5] = b;
            PitchRoll2AB(mPt.positions[16], mPt.positions[17], &a, &b);
            mPt.positions[16] = -a;
            mPt.positions[17] = -b;
        }

        for (int i = 0; i < mNumAction; ++i)
            mPt.positions[i] = zmq_clamp_target(mPt.positions[i], i);

        mPt.time_from_start = rclcpp::Duration(std::chrono::duration<double>(mTimeStep));
    }

    static void fsm_mimic_jaka_mini_zmq_judge()
    {
        mRobot->get_joy_robot_state(zmq_joydata);
        if (mErrCode != ERR_NONE)
        {
            ReqState = FSM_FAULT;
        }
        else
        {
            if (zmq_joydata.joy.buttons_A)
            {
                mRobot->do_enable(false);
                ReqState = FSM_FAULT;
            }
            else if (mNoTargetEnabled && zmq_joydata.joy.buttons_LB && zmq_joydata.joy.buttons_B)
            {
                ReqState = FSM_NOTARGET_HOLD;
            }

            if (!zmq_play && zmq_joydata.joy.buttons_RB)
            {
                zmq_play = true;
            }
        }

        if (ReqState != FSM_JAKAMINIMIMIC_ZMQ)
        {
            from_fsm_mimic_jaka_mini_zmq();
        }
    }

    static void fsm_mimic_jaka_mini_zmq_run()
    {
        if (!zmq_fetch_robot_state())
        {
            mErrCode = ERR_JOINT;
            return;
        }

        if (zmq_first_step)
        {
            auto frames = zmq_motion_buffer->get_obs();
            zmq_initialize_alignment(frames[0]);
            zmq_initialize_last_action();
            zmq_construct_single_obs();
            zmq_init_history();
            zmq_pack_history();
            zmq_first_step = false;
        }
        else if (!zmq_last_ready && zmq_motion_buffer->ready())
        {
            // Align again on first ZMQ package arrival

            printf("-------------last_ready-----------\n");
            auto frames = zmq_motion_buffer->get_obs();
            zmq_initialize_alignment(frames[0]);
            zmq_construct_single_obs();
            zmq_push_history();
            zmq_pack_history();
        }
        else
        {
            zmq_construct_single_obs();
            zmq_push_history();
            zmq_pack_history();
        }
        zmq_last_ready = zmq_motion_buffer->ready();

        mMimicJakaMiniZmqPolicy.session->Run(
            Ort::RunOptions(nullptr),
            &mMimicJakaMiniZmqPolicy.input_name,
            mMimicJakaMiniZmqPolicy.input_tensor,
            1,
            &mMimicJakaMiniZmqPolicy.output_name,
            mMimicJakaMiniZmqPolicy.output_tensor,
            1);

        zmq_fill_trajectory();
        mRobot->set_joint_trajectory(mJointNames, mPt);
    }

    void FSMMimicJakaMiniZmqEnter(FSMSTATE from)
    {
        if (!zmq_resolve_policy_layout())
        {
            mErrCode = ERR_INIT;
            return;
        }

        mFSMState = ReqState = FSM_JAKAMINIMIMIC_ZMQ;
        StateJudge = &fsm_mimic_jaka_mini_zmq_judge;
        StateRun = &fsm_mimic_jaka_mini_zmq_run;
        zmq_play = false;
        zmq_first_step = true;
        zmq_last_ready = false;

        if (mDebugFlag)
            printf("--------from state %d to jaka mini mimic ZMQ state %d------------\n",
                   from, FSM_JAKAMINIMIMIC_ZMQ);

        if (!zmq_fetch_robot_state())
        {
            mErrCode = ERR_JOINT;
            return;
        }

        zmq_motion_buffer = std::make_unique<RealtimeMotionBuffer>(
            mJointNames,
            mJakaMiniZmqConnect,
            mJakaMiniZmqHwm,
            static_cast<int64_t>(mJakaMiniZmqDt * 1e9),
            static_cast<int64_t>(mJakaMiniZmqTolerance * 1e9),
            mJakaMiniZmqFutureSteps,
            mJakaMiniZmqDefaultJointPos.size() > 0 ? mJakaMiniZmqDefaultJointPos
                                                   : Eigen::VectorXf::Zero(mNumAction),
            mJakaMiniZmqDefaultRootPos,
            mJakaMiniZmqDefaultRootQuat);

        switch (from)
        {
        case FSM_IDLE:
            break;
        case FSM_NOTARGET_HOLD:
            break;
        case FSM_NOTARGET:
            break;
        default:
            mErrCode = ERR_STATE_TRANSITION;
            break;
        }
    }

    static void from_fsm_mimic_jaka_mini_zmq()
    {
        zmq_motion_buffer.reset();
        switch (ReqState)
        {
        case FSM_IDLE:
            FSMIdleEnter(FSM_JAKAMINIMIMIC_ZMQ);
            break;
        case FSM_LOCO:
            FSMLocoEnter(FSM_JAKAMINIMIMIC_ZMQ);
            break;
        case FSM_MIMIC:
            FSMMimicEnter(FSM_JAKAMINIMIMIC_ZMQ);
            break;
        case FSM_NOTARGET_HOLD:
            FSMNoTargetHoldEnter(FSM_JAKAMINIMIMIC_ZMQ);
            break;
        case FSM_NOTARGET:
            FSMNoTargetEnter(FSM_JAKAMINIMIMIC_ZMQ);
            break;
        case FSM_FAULT:
            FSMFaultEnter(FSM_JAKAMINIMIMIC_ZMQ);
            break;
        default:
            mErrCode = ERR_STATE_TRANSITION;
            break;
        }
    }

} // namespace RLController