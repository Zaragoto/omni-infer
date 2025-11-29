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

class Request {
public:
    struct LayerState {
        bool ready = false;
        bool sent = false;
        bool need = false;
        bool sending = false;
        std::vector<boost::asio::mutable_buffer> buffers;

        asio::awaitable<void> send(tcp::socket& socket, int layer_id)
        {
            if (!socket.is_open()) {
                std::cerr << "[P] Socket is not open, cannot send layer " << layer_id << std::endl;
                sending = false;
                co_return;
            }

            if (buffers.empty()) {
                std::cerr << "[P] layer " << layer_id << " buffers empty, skip send" << std::endl;
                sending = false;
                co_return;
            }

            std::cerr << "[P][LayerState] Sending layer " << layer_id << " with " << buffers.size()
                    << " chunks" << std::endl;

            try {
                // 使用分散写入所有buffer
                std::size_t total_written = 0;
                for (const auto& buffer : buffers) {
                    std::size_t n = co_await asio::async_write(socket, 
                        asio::buffer(buffer.data(), buffer.size()), 
                        asio::use_awaitable);
                    total_written += n;
                }

                std::cerr << "[P] layer "<< layer_id <<" async_write SUCCESS, total bytes sent = " << total_written << std::endl;

                sent = true;
                sending = false;
                need = false;
                
            } catch (const boost::system::system_error &e) {
                sending = false;
                std::cerr << "[P] layer "<< layer_id <<" async_write FAILED: " << e.what() << std::endl;
            } catch (const std::exception &e) {
                sending = false;
                std::cerr << "[P] layer "<< layer_id <<" async_write FAILED with exception: " << e.what() << std::endl;
            }
            co_return; 
        }
    };

    int64_t uid;
    std::map<int, LayerState> layer_states;
    std::mutex req_mutex;

    Request(int64_t uid_, int num_layers)
        : uid(uid_)
    {
        for (int i = 0; i < num_layers; i++) {
            layer_states.emplace(i, LayerState());
        }
    }
};

inline std::unordered_map<int64_t, std::shared_ptr<Request>> global_requests;
inline std::mutex global_requests_mutex;

inline std::shared_ptr<Request> find_request(int64_t uid) {
    std::lock_guard<std::mutex> lock(global_requests_mutex);
    auto it = global_requests.find(uid);
    return (it != global_requests.end()) ? it->second : nullptr;
}

inline std::shared_ptr<Request> ensure_request(int64_t uid, int num_layers)
{
    std::lock_guard<std::mutex> lock(global_requests_mutex);
    auto it = global_requests.find(uid);
    if (it != global_requests.end()) {
        return it->second;
    }
    auto req = std::make_shared<Request>(uid, num_layers);
    global_requests[uid] = req;
    return req;
}

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