#include "RealtimeMotionBuffer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <zmq.h>
#include <zmq_addon.hpp>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace RLController
{

    namespace
    {
        Eigen::Quaternionf normalize_quat(const Eigen::Quaternionf &q)
        {
            if (q.norm() < 1.0e-6f)
            {
                return Eigen::Quaternionf::Identity();
            }
            Eigen::Quaternionf r = q;
            r.normalize();
            return r;
        }

        Eigen::Quaternionf slerp_quat(const Eigen::Quaternionf &q0_in,
                                      const Eigen::Quaternionf &q1_in,
                                      float t)
        {
            return normalize_quat(q0_in).slerp(t, normalize_quat(q1_in));
        }
    } // namespace

    RealtimeMotionBuffer::RealtimeMotionBuffer(
        const std::vector<std::string> &joint_names,
        const std::string &zmq_endpoint,
        int zmq_hwm,
        int64_t dt_ns,
        int64_t tolerance_ns,
        const std::vector<int> &future_steps,
        const Eigen::VectorXf &default_joint_pos,
        const Eigen::Vector3f &default_root_pos,
        const Eigen::Quaternionf &default_root_quat)
        : _num_joints(static_cast<int>(joint_names.size())),
          _dt_ns(dt_ns),
          _tolerance_ns(tolerance_ns),
           _future_steps(future_steps),
           _zmq_context(nullptr),
           _zmq_socket(nullptr),
           _zmq_endpoint(zmq_endpoint),
           _default_joint_pos(default_joint_pos),
           _default_root_pos(default_root_pos),
           _default_root_quat(normalize_quat(default_root_quat)),
           _identity_quat(Eigen::Quaternionf::Identity())
    {
        if (dt_ns <= 0)
        {
            throw std::invalid_argument("dt_ns must be positive");
        }
        if (static_cast<int>(_future_steps.size()) != kFutureSteps)
        {
            throw std::invalid_argument("future_steps must have exactly " +
                                        std::to_string(kFutureSteps) + " elements");
        }
        if (_num_joints <= 0)
        {
            throw std::invalid_argument("joint_names must not be empty");
        }

        _min_future_step = *std::min_element(_future_steps.begin(), _future_steps.end());
        _max_future_step = *std::max_element(_future_steps.begin(), _future_steps.end());

        for (int i = 0; i < kFutureSteps; ++i)
        {
            _future_steps_ns[i] = static_cast<int64_t>(_future_steps[i]) * _dt_ns;
        }

        _delay_ns = static_cast<int64_t>(_max_future_step) * _dt_ns + _tolerance_ns;
        _history_ns = _delay_ns + std::abs(_min_future_step) * _dt_ns;

        _running = true;

        if (!_zmq_endpoint.empty())
        {
            _zmq_context = zmq_ctx_new();
            if (!_zmq_context)
            {
                throw std::runtime_error("Failed to create ZMQ context");
            }

            _zmq_socket = zmq_socket(_zmq_context, ZMQ_SUB);
            if (!_zmq_socket)
            {
                zmq_ctx_destroy(_zmq_context);
                _zmq_context = nullptr;
                throw std::runtime_error("Failed to create ZMQ SUB socket");
            }

            int linger = 0;
            zmq_setsockopt(_zmq_socket, ZMQ_LINGER, &linger, sizeof(linger));
            zmq_setsockopt(_zmq_socket, ZMQ_RCVHWM, &zmq_hwm, sizeof(zmq_hwm));
            zmq_setsockopt(_zmq_socket, ZMQ_SUBSCRIBE, "", 0);

            int rc = zmq_connect(_zmq_socket, _zmq_endpoint.c_str());
            if (rc != 0)
            {
                std::fprintf(stderr, "[RealtimeMotionBuffer] ZMQ connect to %s failed: %s\n",
                             _zmq_endpoint.c_str(), zmq_strerror(zmq_errno()));
            }
            else
            {
                std::printf("[RealtimeMotionBuffer] ZMQ connected to %s\n", _zmq_endpoint.c_str());
            }

            _zmq_thread = std::thread(&RealtimeMotionBuffer::zmq_thread_loop, this);
        }
    }

    RealtimeMotionBuffer::~RealtimeMotionBuffer()
    {
        _running = false;
        if (_zmq_thread.joinable())
        {
            _zmq_thread.join();
        }
        if (_zmq_socket)
        {
            zmq_close(_zmq_socket);
            _zmq_socket = nullptr;
        }
        if (_zmq_context)
        {
            zmq_ctx_destroy(_zmq_context);
            _zmq_context = nullptr;
        }
    }

    bool RealtimeMotionBuffer::ready() const
    {
        std::lock_guard<std::mutex> lock(_lock);
        return !_timestamps_ns.empty();
    }

    void RealtimeMotionBuffer::zmq_thread_loop()
    {
        while (_running)
        {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            int rc = zmq_msg_recv(&msg, _zmq_socket, ZMQ_DONTWAIT);
            if (rc < 0)
            {
                zmq_msg_close(&msg);
                if (zmq_errno() == EAGAIN)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            try
            {
                std::string json_str(static_cast<const char *>(zmq_msg_data(&msg)),
                                     zmq_msg_size(&msg));
                zmq_msg_close(&msg);
                append_payload(json_str);
            }
            catch (const std::exception &e)
            {
                zmq_msg_close(&msg);
                std::fprintf(stderr, "[RealtimeMotionBuffer] append_payload error: %s\n", e.what());
            }
        }
    }

    void RealtimeMotionBuffer::append_payload(const std::string &json_str)
    {
        json payload = json::parse(json_str);
        if (!payload.is_object())
        {
            throw std::runtime_error("Payload is not a JSON object");
        }

        int64_t timestamp_ns = 0;
        timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();

        json joint_pos_json;
        bool is_single_key = false;
        if (payload.contains("joint_pos"))
        {
            joint_pos_json = payload["joint_pos"];
            is_single_key = true;
        }
        else if (payload.contains("dof_pos"))
        {
            joint_pos_json = payload["dof_pos"];
        }
        else if (payload.contains("qpos"))
        {
            joint_pos_json = payload["qpos"];
        }
        else
        {
            throw std::runtime_error("Payload missing joint_pos/dof_pos/qpos");
        }

        std::vector<float> joint_pos_vec = joint_pos_json.get<std::vector<float>>();
        if (!is_single_key && static_cast<int>(joint_pos_vec.size()) >= 7 + _num_joints)
        {
            joint_pos_vec.erase(joint_pos_vec.begin(), joint_pos_vec.begin() + 7);
        }
        if (static_cast<int>(joint_pos_vec.size()) != _num_joints)
        {
            throw std::runtime_error("Expected " + std::to_string(_num_joints) +
                                     " joint positions, got " + std::to_string(joint_pos_vec.size()));
        }

        if (!payload.contains("body_pos_w") || !payload.contains("body_quat_w"))
        {
            throw std::runtime_error("Payload missing body_pos_w/body_quat_w");
        }

        auto body_pos_w = payload["body_pos_w"].get<std::vector<std::vector<float>>>();
        auto body_quat_w = payload["body_quat_w"].get<std::vector<std::vector<float>>>();

        if (body_pos_w.empty() || body_quat_w.empty())
        {
            throw std::runtime_error("body_pos_w or body_quat_w is empty");
        }

        // Hard-coded as 3 which is waist_yaw_link in Isaac Lab order
        // const auto &root_pos_vec = body_pos_w[13];
        // const auto &root_quat_vec = body_quat_w[13];
        const auto &root_pos_vec = body_pos_w[0];
        const auto &root_quat_vec = body_quat_w[0];

        if (root_pos_vec.size() < 3 || root_quat_vec.size() < 4)
        {
            throw std::runtime_error("Root body pos/quat has wrong dimension");
        }

        Eigen::Vector3f root_pos(root_pos_vec[0], root_pos_vec[1], root_pos_vec[2]);
        Eigen::Quaternionf root_quat(root_quat_vec[0], root_quat_vec[1],
                                     root_quat_vec[2], root_quat_vec[3]);
        root_quat = normalize_quat(root_quat);

        Eigen::VectorXf joint_pos = Eigen::VectorXf::Map(joint_pos_vec.data(),
                                                         static_cast<Eigen::Index>(joint_pos_vec.size()));

        // Reorder from MuJoCo joint order to Isaac Lab joint order.
        static constexpr std::array<int, 27> kMujocoToIsaacIndex = {
            0, 6, 12, 1, 7, 13, 25, 19, 2, 8, 14, 26, 20,
            3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23, 18, 24};
        if (static_cast<size_t>(joint_pos.size()) == kMujocoToIsaacIndex.size())
        {
            Eigen::VectorXf reordered(joint_pos.size());
            for (Eigen::Index m = 0; m < joint_pos.size(); ++m)
                reordered[m] = joint_pos[static_cast<Eigen::Index>(kMujocoToIsaacIndex[m])];
            joint_pos = reordered;
        }

        std::lock_guard<std::mutex> lock(_lock);

        if (_timestamps_ns.empty() || timestamp_ns >= _timestamps_ns.back())
        {
            _timestamps_ns.push_back(timestamp_ns);
            _root_pos_frames.push_back(root_pos);
            _root_quat_frames.push_back(root_quat);
            _joint_pos_frames.push_back(joint_pos);
        }
        else
        {
            auto it = std::upper_bound(_timestamps_ns.begin(), _timestamps_ns.end(), timestamp_ns);
            auto idx = std::distance(_timestamps_ns.begin(), it);
            _timestamps_ns.insert(it, timestamp_ns);
            _root_pos_frames.insert(_root_pos_frames.begin() + idx, root_pos);
            _root_quat_frames.insert(_root_quat_frames.begin() + idx, root_quat);
            _joint_pos_frames.insert(_joint_pos_frames.begin() + idx, joint_pos);
        }
    }

    void RealtimeMotionBuffer::cleanup(int64_t cutoff_ns)
    {
        while (_timestamps_ns.size() > 1 && _timestamps_ns[1] < cutoff_ns)
        {
            _timestamps_ns.erase(_timestamps_ns.begin());
            _root_pos_frames.erase(_root_pos_frames.begin());
            _root_quat_frames.erase(_root_quat_frames.begin());
            _joint_pos_frames.erase(_joint_pos_frames.begin());
        }
    }

    void RealtimeMotionBuffer::fill_sample_frames(
        const std::array<int64_t, kFutureSteps> &target_times_ns,
        std::array<InterpolatedFrame, kFutureSteps> &out) const
    {
        if (_timestamps_ns.empty())
        {
            for (int i = 0; i < kFutureSteps; ++i)
            {
                out[i].root_pos_w = _default_root_pos;
                out[i].root_quat_w = _default_root_quat;
                out[i].dof_pos = _default_joint_pos;
            }
            return;
        }

        if (_timestamps_ns.size() == 1)
        {
            for (int i = 0; i < kFutureSteps; ++i)
            {
                out[i].root_pos_w = _root_pos_frames[0];
                out[i].root_quat_w = _root_quat_frames[0];
                out[i].dof_pos = _joint_pos_frames[0];
            }
            return;
        }

        const int64_t t_min = _timestamps_ns.front();
        const int64_t t_max = _timestamps_ns.back();
        const size_t n = _timestamps_ns.size();

        for (int step = 0; step < kFutureSteps; ++step)
        {
            int64_t t = target_times_ns[step];
            t = std::max(t_min, std::min(t_max, t));

            auto it = std::upper_bound(_timestamps_ns.begin(), _timestamps_ns.end(), t);
            size_t right_idx = static_cast<size_t>(std::distance(_timestamps_ns.begin(), it));
            if (right_idx >= n)
                right_idx = n - 1;
            if (right_idx == 0)
                right_idx = 1;
            size_t left_idx = right_idx - 1;

            int64_t t0 = _timestamps_ns[left_idx];
            int64_t t1 = _timestamps_ns[right_idx];
            float alpha = (t1 > t0) ? static_cast<float>(t - t0) / static_cast<float>(t1 - t0) : 0.0f;

            out[step].root_pos_w = _root_pos_frames[left_idx] +
                                   alpha * (_root_pos_frames[right_idx] - _root_pos_frames[left_idx]);
            out[step].root_quat_w = slerp_quat(_root_quat_frames[left_idx],
                                               _root_quat_frames[right_idx], alpha);
            out[step].dof_pos = _joint_pos_frames[left_idx] +
                                alpha * (_joint_pos_frames[right_idx] - _joint_pos_frames[left_idx]);
        }
    }

    std::array<InterpolatedFrame, RealtimeMotionBuffer::kFutureSteps> RealtimeMotionBuffer::get_obs()
    {
        int64_t current_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();

        int64_t cutoff_ns = current_time_ns - _history_ns;

        {
            std::lock_guard<std::mutex> lock(_lock);
            cleanup(cutoff_ns);
        }

        int64_t target_base_ns = current_time_ns - _delay_ns;

        std::array<int64_t, kFutureSteps> target_times_ns{};
        for (int i = 0; i < kFutureSteps; ++i)
        {
            target_times_ns[i] = target_base_ns + _future_steps_ns[i];
        }

        std::array<InterpolatedFrame, kFutureSteps> result{};
        {
            std::lock_guard<std::mutex> lock(_lock);
            fill_sample_frames(target_times_ns, result);
        }
        return result;
    }

    int64_t RealtimeMotionBuffer::latest_timestamp_ns() const
    {
        std::lock_guard<std::mutex> lock(_lock);
        if (_timestamps_ns.empty())
            return 0;
        return _timestamps_ns.back();
    }

} // namespace RLController
