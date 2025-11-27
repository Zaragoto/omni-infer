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

// Request: For each request, the src_id_list and dst_id_list calculate a specific UID.
class Request {
public:
    struct LayerState {
        bool ready = false;
        bool sent = false;
        bool need = false;
        bool sending = false;
        std::string last_ready_modifier = "默认值";
        std::string last_sent_modifier = "默认值";
        std::string last_need_modifier = "默认值";
        std::string last_sending_modifier = "默认值";
        std::vector<boost::asio::mutable_buffer> buffers;

        LayerState() : ready(false), sent(false) {
            last_ready_modifier = "无参构造函数";
        }

        asio::awaitable<void> send(tcp::socket& socket, int layer_id)
        {
            if (!socket.is_open()) {
                std::cerr << "[P] Socket is not open, cannot send." << std::endl;
                sending = false;
                last_sending_modifier = "第" + std::to_string(layer_id) + "层发送时socket没开后更新";
                co_return;
            }

            if (buffers.empty()) {
                std::cerr << "[P] layer " << layer_id
                          << " buffers empty, skip send" << std::endl;
                sending = false;
                last_sending_modifier = "第" + std::to_string(layer_id) + "层发送时buffers为空更新";
                co_return;
            }

            std::cerr << "[P][LayerState] Sending layer with " << buffers.size()
                    << " chunks, first 8 bytes = " << *(int64_t *)buffers[0].data() << std::endl;

            try {
                std::size_t n = co_await asio::async_write(socket, buffers, asio::use_awaitable);
                std::cerr << "[P] layer "<< layer_id <<" async_write SUCCESS, bytes sent = " << n << std::endl;

                // 发送成功：标记已发送完成，并清除 sending 标志
                sent = true;
                last_sent_modifier = "第" + std::to_string(layer_id) + "层发送成功后更新";
                sending = false;
                last_sending_modifier = "第" + std::to_string(layer_id) + "层发送成功后更新";
                need = false;
                last_need_modifier = "第" + std::to_string(layer_id) + "层发送成功后更新";
                std::cerr << "已发送的层 " << layer_id << "的状态已变为: " << std::endl
                                          << "ready = " << ready
                                          << ", sent = " << sent
                                          << ", need = " << need
                                          << ", sending = " << sending << std::endl
                                          << "last_ready_modifier = " << last_ready_modifier << std::endl
                                          << "last_sent_modifier = " << last_sent_modifier << std::endl
                                          << "last_need_modifier = " << last_need_modifier << std::endl
                                          << "last_sending_modifier = " << last_sending_modifier << std::endl;
                                          
            } catch (const boost::system::system_error &e) {
                sending = false;  // 失败也要清掉，否则外层永远以为“正在发送中”
                last_sending_modifier = "第" + std::to_string(layer_id) + "层发送失败抛出boost::system::system_error后更新";
                auto error_code = e.code();
                std::cerr << "[P] layer "<< layer_id <<" async_write FAILED with system_error: "
                          << error_code.message() << " 其状态现在为: " << std::endl
                          << "ready = " << ready
                          << ", sent = " << sent
                          << ", need = " << need
                          << ", sending = " << sending << std::endl
                          << "last_ready_modifier = " << last_ready_modifier << std::endl
                          << "last_sent_modifier = " << last_sent_modifier << std::endl
                          << "last_need_modifier = " << last_need_modifier << std::endl
                          << "last_sending_modifier = " << last_sending_modifier << std::endl;
                          
                if (error_code == asio::error::eof) {
                    std::cerr << "Connection closed by peer. 其状态现在为: " << std::endl
                              << "ready = " << ready
                              << ", sent = " << sent
                              << ", need = " << need
                              << ", sending = " << sending << std::endl
                              << "last_ready_modifier = " << last_ready_modifier << std::endl
                              << "last_sent_modifier = " << last_sent_modifier << std::endl
                              << "last_need_modifier = " << last_need_modifier << std::endl
                              << "last_sending_modifier = " << last_sending_modifier << std::endl;
                              
                } else if (error_code == asio::error::connection_reset) {
                    std::cerr << "Connection reset by peer. 其状态现在为: " << std::endl
                              << "ready = " << ready
                              << ", sent = " << sent
                              << ", need = " << need
                              << ", sending = " << sending << std::endl
                              << "last_ready_modifier = " << last_ready_modifier << std::endl
                              << "last_sent_modifier = " << last_sent_modifier << std::endl
                              << "last_need_modifier = " << last_need_modifier << std::endl
                              << "last_sending_modifier = " << last_sending_modifier << std::endl;
                              
                }
            } catch (const std::exception &e) {
                sending = false;
                last_sending_modifier = "第" + std::to_string(layer_id) + "层发送失败抛出std::exception后更新";
                std::cerr << "[P] layer "<< layer_id <<" async_write FAILED with exception: "
                        << e.what() << " 其状态现在为：" << std::endl
                        << "ready = " << ready
                        << ", sent = " << sent
                        << ", need = " << need
                        << ", sending = " << sending << std::endl
                        << "last_ready_modifier = " << last_ready_modifier << std::endl
                        << "last_sent_modifier = " << last_sent_modifier << std::endl
                        << "last_need_modifier = " << last_need_modifier << std::endl
                        << "last_sending_modifier = " << last_sending_modifier << std::endl;

            } catch (...) {
                sending = false;
                last_sending_modifier = "第" + std::to_string(layer_id) + "层发送失败抛出其他异常后更新";
                std::cerr << "[P] layer " << layer_id
                          << " async_write FAILED (unknown). 其状态现在为：" << std::endl
                          << "ready = " << ready
                          << ", sent = " << sent
                          << ", need = " << need
                          << ", sending = " << sending << std::endl
                          << "last_ready_modifier = " << last_ready_modifier << std::endl
                          << "last_sent_modifier = " << last_sent_modifier << std::endl
                          << "last_need_modifier = " << last_need_modifier << std::endl
                          << "last_sending_modifier = " << last_sending_modifier << std::endl;
                          
            }
            co_return; 
        }
    };

    int64_t uid;

    // key = layer_id -> state
    std::map<int, LayerState> layer_states;

    // mutex for one request
    std::mutex req_mutex;

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

inline std::shared_ptr<Request> find_request(int64_t uid) {
    std::lock_guard<std::mutex> lock(global_requests_mutex);

    auto it = global_requests.find(uid);
    return (it != global_requests.end()) ? it->second : nullptr;
}

inline bool is_layer_ready(std::shared_ptr<Request> req, int layer)
{
    std::lock_guard<std::mutex> lock(req->req_mutex);
    return req->layer_states[layer].ready;
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