#pragma once
#include <unordered_map>
#include <memory>
#include <mutex>
#include <map>
#include <string>
#include <vector>
#include <boost/asio.hpp>
#include <ox_block_table.hpp>
#include <system_error>

using boost::asio::ip::tcp;

class Session;

enum class LayerPhase : uint8_t {
    Init = 0, // Initialization: No data, no request.
    Need,     // D-side has requested data, but ZMQ transmission has not yet arrived.
    Ready,    // ZMQ transmission has arrived, but the D-side has not yet requested it.
    Pending,  // ZMQ transmission has arrived, and the D-side has issued an application, allowing it to join the queue.
    Enqueued, // The data has been sent to send_queue_, awaiting processing by the sending coroutine.
    Sending,  // async_writing
    Done,     // Sent successfully
    Failed    // Sent failed, waiting for retransmission
};

// Request: For each request, the src_id_list and dst_id_list calculate a specific UID.
class Request {
public:
    struct LayerState {
        LayerPhase phase = LayerPhase::Init;
        std::string last_phase_modifier = "Init 默认";
        std::vector<boost::asio::mutable_buffer> buffers;

        LayerState() = default;

        bool is_done() const noexcept {
            return phase == LayerPhase::Done;
        }

        bool can_enqueue() const noexcept {
            return phase == LayerPhase::Pending;
        }

        const char* phase_str() const noexcept {
            switch(phase) {
            case LayerPhase::Init: return "Init";
            case LayerPhase::Need: return "Need";
            case LayerPhase::Ready: return "Ready";
            case LayerPhase::Pending: return "Pending";
            case LayerPhase::Enqueued: return "Enqueued";
            case LayerPhase::Sending: return "Sending";
            case LayerPhase::Done: return "Done";
            case LayerPhase::Failed: return "Failed";
            }
            return "Unknown";
        }

        void log_phase(const char* tag, int layer_id) const {
            std::cerr << "[P][LAYER][" << tag << "] layer = " << layer_id
                      << " phase = " << phase_str()
                      << " note = " << last_phase_modifier
                      << " buffers size = " << buffers.size()
                      << std::endl;
        }

        void after_receiving_zmq(int layer_id, const std::string& reason) {
            switch(phase) {
            case LayerPhase::Init:
                phase = LayerPhase::Ready;
                last_phase_modifier = std::string("ZMQ arrived: ") + reason;
                break;
            case LayerPhase::Need:
            case LayerPhase::Failed:
                phase = LayerPhase::Pending;
                last_phase_modifier = std::string("ZMQ arrived(requested or failed)") + reason;
                break;
            case LayerPhase::Ready:
            case LayerPhase::Pending:
            case LayerPhase::Enqueued:
            case LayerPhase::Sending:
            case LayerPhase::Done:
                last_phase_modifier = std::string("ZMQ repeated notification") + reason;
                break;
            }
            log_phase("DATA READY", layer_id);
        }

        void after_receiving_tcp(int layer_id, const std::string& reason) {
            switch(phase) {
            case LayerPhase::Init:
                phase = LayerPhase::Need;
                last_phase_modifier = std::string("TCP uid arrived: ") + reason;
                break;
            case LayerPhase::Ready:
            case LayerPhase::Failed:
                phase = LayerPhase::Pending;
                last_phase_modifier = std::string("TCP arrived(data ready or failed)") + reason;
                break;
            case LayerPhase::Need:
            case LayerPhase::Pending:
            case LayerPhase::Enqueued:
            case LayerPhase::Sending:
            case LayerPhase::Done:
                last_phase_modifier = std::string("TCP repeated notification") + reason;
                break;
            }
            log_phase("DATA REQUESTED", layer_id);
        }

        void mark_enqueued(int layer_id, const std::string& reason) {
            phase = LayerPhase::Enqueued;
            last_phase_modifier = std::string("sent to send_queue: ") + reason;
            log_phase("MARK ENQUEUED", layer_id);
        }

        void mark_sending(int layer_id, const std::string& reason) {
            phase = LayerPhase::Sending;
            last_phase_modifier = std::string("ready for sending: ") + reason;
            log_phase("MARK SENDING", layer_id);
        }

        void mark_done(int layer_id, const std::string& reason) {
            phase = LayerPhase::Done;
            last_phase_modifier = std::string("sent successfully: ") + reason;
            log_phase("mark_done", layer_id);
        }

        void mark_failed(int layer_id, const std::string& reason) {
            phase = LayerPhase::Failed;
            last_phase_modifier = std::string("sent failed: ") + reason;
            log_phase("MARK FAILED", layer_id);
        }

        asio::awaitable<void> send(tcp::socket& socket, int layer_id)
        {
            log_phase("SEND ENTER", layer_id);

            if (!socket.is_open()) {
                mark_failed(layer_id, "socket is not open");
                co_return;
            }

            if (buffers.empty()) {
                mark_failed(layer_id, "empty buffers");
                co_return;
            }

            std::cerr << "[P][LayerState] Sending layer with " << buffers.size()
                      << " chunks, first 8 bytes = " << *(int64_t *)buffers[0].data() << std::endl;

            try {
                /*
                asio::steady_timer timer(co_await asio::this_coro::executor);
                timer.expires_after(std::chrono::milliseconds(500));

                auto write_op   = asio::async_write(socket, buffers, asio::use_awaitable);
                auto timeout_op = timer.async_wait(asio::use_awaitable);

                auto result = co_await (std::move(write_op) || std::move(timeout_op));

                if (result.index() == 1) {   // 超时
                    // socket.cancel();
                    boost::system::error_code ec;
                    socket.cancel(ec);
                    if(ec) {
                        std::cerr << "[P][TIMEOUT] cancel error: " << ec.message() << std::endl;
                    }

                    throw std::runtime_error("async_write timeout");
                }

                // 从 result 中取回 n
                std::size_t n = std::get<0>(result);
                */

                std::size_t n = co_await asio::async_write(socket, buffers, asio::use_awaitable);

                std::size_t p_send_total = 0;
                for (auto& b : buffers)
                    p_send_total += boost::asio::buffer_size(b);

                std::cerr << "[P][SEND_CHECK] uid=? layer=" << layer_id
                          << " buffers_total=" << p_send_total
                          << " bytes_sent = " << n
                          << std::endl;
                mark_done(layer_id, "async_write SUCCESS");
            } catch (const boost::system::system_error &e) {
                auto error_code = e.code();
                mark_failed(layer_id, std::string("boost::system::system_error: ") + error_code.message());
            } catch (const std::exception &e) {
                mark_failed(layer_id, std::string("std::exception: ") + e.what());

            } catch (...) {
                mark_failed(layer_id, "unknown exception");
            }

            log_phase("SEND EXIT", layer_id);
            co_return;
        }

    };

    int64_t uid;
    std::map<int, LayerState> layer_states;
    std::mutex req_mutex;
    std::weak_ptr<Session> owner;

    Request(int64_t uid_, int num_layers)
        : uid(uid_)
    {
        for (int i = 0; i < num_layers; i++) {
            layer_states.emplace(i, LayerState());
        }
    }
};


// Global request map
inline std::unordered_map<int64_t, std::shared_ptr<Request>> global_requests;
inline std::mutex global_requests_mutex;

inline void add_request(std::shared_ptr<Request> req) {
    if (!req) return;

    std::lock_guard<std::mutex> lock(global_requests_mutex);
    global_requests[req->uid] = req;
}

inline void remove_request(int64_t uid) {
    std::lock_guard<std::mutex> lock(global_requests_mutex);

    global_requests.erase(uid);
}

inline int64_t generate_uid_from_block_list(const block_list_t& block_ids) {
    const uint64_t FNV_OFFSET = 1469598103934665603ULL;
    const uint64_t FNV_PRIME  = 1099511628211ULL;

    uint64_t hash = FNV_OFFSET;

    for (auto id : block_ids) {
        uint64_t v = static_cast<uint64_t>(id);
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (v >> (i * 8)) & 0xff;
            hash ^= byte;
            hash *= FNV_PRIME;
        }
    }

    return static_cast<int64_t>(hash);
}