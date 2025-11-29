// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <set>
#include <atomic>
#include <mutex>
#include <exception>
#include <msgpack.hpp>
#include <sys/mman.h>
#include <shared_mutex>

#include <ox_config.hpp>
#include <zmq_coroutine.hpp>
#include <ox_metrics.hpp>
#include <ox_server.hpp>
#include <ox_log.hpp>
#include <ox_kv_merger.h>
#include <ox_request_manager.hpp>
#include <chrono>
#include <boost/asio/experimental/parallel_group.hpp>
#include <boost/asio/experimental/promise.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;
using namespace boost::asio::experimental::awaitable_operators;

struct RequestMessage {
    request_id_t request_id;
    table_id_t table_id;
    block_list_t src_block_ids;
    block_list_t dst_block_ids;
    int cluster_id{0};
    MSGPACK_DEFINE_MAP(request_id, table_id, src_block_ids, dst_block_ids, cluster_id)
};

struct ResponseMessage {
    request_id_t request_id;
    int layer_id;
    bool success;
    MSGPACK_DEFINE_MAP(request_id, layer_id, success)
};

using ResponseTask = std::tuple<client_id_t, request_id_t, int, bool>;
using ZMQChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ResponseTask)>;

using ConnectionMessage = std::tuple<request_id_t, table_id_t, block_list_t, block_list_t>;
using ConnectionChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ConnectionMessage)>;

using ShardMessage = std::tuple<std::string, block_list_t>;
using ShardChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ShardMessage)>;

using GroupMessage = std::tuple<client_id_t, int>;
using GroupChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, GroupMessage)>;

class CoroutineConnection {
public:
    CoroutineConnection(asio::io_context &io_context, boost::asio::ip::tcp::endpoint addr, Config &config, int rank,
        BlockTable &bt, ShardChannel &response)
        : socket(io_context), config(config), rank(rank), addr(addr), bt(bt), request(config.get_io_context(), 128),
          upstream(response)
    {}

    void start()
    {
        co_spawn(socket.get_executor(), run(), asio::detached);
    }

    asio::awaitable<void> run()
    {
        try {
            co_await socket.async_connect(addr, asio::use_awaitable);
            optimize_tcp_socket(socket);
            std::cerr << "[D_SIDE_CONN] INFO: Successfully connected to P-side shard at " << addr.address() << ":"
                      << addr.port() << "\n";

            while (true) {
                auto [request_id, table_id, src_ids, dst_ids] = co_await request.async_receive(asio::use_awaitable);
                std::cerr << "[D] Processing request " << request_id << std::endl;

                if (src_ids.empty()) {
                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
                    continue;
                }

                auto bufs = bt.get_buffers(table_id, dst_ids, rank);
                
                int64_t uid = generate_uid_from_block_list(src_ids);
                
                try {
                    // 简单直接的数据传输
                    co_await asio::async_write(socket, asio::buffer(&uid, sizeof(uid)), asio::use_awaitable);
                    co_await asio::async_read(socket, bufs, asio::use_awaitable);
                    
                    std::cerr << "[D] Successfully received data for request " << request_id << std::endl;

                } catch (const std::exception &e) {
                    std::cerr << "[D] Data transfer failed: " << e.what() << std::endl;
                }
                
                global_stats_update(dst_ids.size() * bt.block_tp_size());

                co_await upstream.async_send(
                    boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
            }
        } catch (const std::exception &e) {
            std::cerr << "Connection error: " << e.what() << "\n";
        }
    }

    asio::awaitable<void> submit_request(
        const std::string &request_id, table_id_t table_id, const block_list_t &src_block_ids, const block_list_t &dst_block_ids)
    {
        co_await request.async_send(boost::system::error_code{},
            std::make_tuple(request_id, table_id, src_block_ids, dst_block_ids),
            asio::use_awaitable);
    }

private:
    tcp::socket socket;
    const Config &config;
    int rank;
    boost::asio::ip::tcp::endpoint addr;

    BlockTable &bt;
    ConnectionChannel request;
    ShardChannel &upstream;
};

class TPShard {
public:
    TPShard(boost::asio::ip::tcp::endpoint addr, int rank, Config &config, BlockTable &bt, GroupChannel &channel)
        : ip(addr), rank(rank), downstream(config.get_io_context(), 128), upstream(channel)
    {
        // 连接建立代码保持不变
        for (std::size_t i = 0; i < config.connections_per_shard; ++i) {
            connections.emplace_back(
                std::make_shared<CoroutineConnection>(config.get_io_context(), ip, config, rank, bt, downstream));
            connections[i]->start();
        }
    }

    asio::awaitable<void> gather(RequestMessage &req)
    {
        // 简化的任务分发逻辑
        size_t total_ids = req.dst_block_ids.size();
        size_t num_conns = std::min(connections.size(), total_ids);

        for (size_t i = 0; i < num_conns; i++) {
            size_t count = total_ids / num_conns + (i < total_ids % num_conns ? 1 : 0);
            if (count == 0) break;

            block_list_t src_ids(req.src_block_ids.begin() + i * count, 
                                req.src_block_ids.begin() + (i + 1) * count);
            block_list_t dst_ids(req.dst_block_ids.begin() + i * count, 
                                req.dst_block_ids.begin() + (i + 1) * count);

            co_spawn(co_await asio::this_coro::executor,
                connections[i]->submit_request(req.request_id, req.table_id, src_ids, dst_ids),
                detached);
        }
    }

    asio::awaitable<void> run()
    {
        while (true) {
            auto message = co_await downstream.async_receive(asio::use_awaitable);
            auto request_id = std::get<0>(message);
            auto ids = std::get<1>(message);

            // 简化的完成通知
            co_await upstream.async_send(
                boost::system::error_code{}, std::make_tuple(request_id, rank), asio::use_awaitable);
        }
    }

private:
    boost::asio::ip::tcp::endpoint ip;
    int rank;
    std::vector<std::shared_ptr<CoroutineConnection>> connections;
    ShardChannel downstream;
    GroupChannel &upstream;
};

class TPGroup {
public:
    TPGroup(Config &config, BlockTable &bt, ZMQChannel &channel)
        : bt(bt), downstream(config.get_io_context(), 128), upstream(channel)  // 修复：初始化 downstream
    {
        // 简化的集群初始化
        if (!config.shard_clusters.empty()) {
            for (size_t c = 0; c < config.shard_clusters.size(); ++c) {
                clusters.emplace_back();
                for (size_t j = 0; j < config.shard_clusters[c].size(); ++j) {
                    auto &ep = config.shard_clusters[c][j];
                    auto shard = std::make_shared<TPShard>(ep, static_cast<int>(j), config, bt, downstream);
                    clusters.back().push_back(shard);
                    co_spawn(config.get_io_context(), shard->run(), detached);
                }
            }
        } else {
            clusters.emplace_back();
            for (int rank = 0; rank < static_cast<int>(config.shard_list.size()); rank++) {
                auto &ep = config.shard_list[rank];
                auto shard = std::make_shared<TPShard>(ep, rank, config, bt, downstream);
                clusters.back().push_back(shard);
                co_spawn(config.get_io_context(), shard->run(), detached);
            }
        }
    }

    asio::awaitable<void> run()
    {
        while (true) {
            auto message = co_await downstream.async_receive(asio::use_awaitable);
            auto request_id = std::get<0>(message);
            // 修复：移除未使用的 rank 变量
            // auto rank = std::get<1>(message);

            // 简化的完成处理
            std::lock_guard<std::mutex> lock(requests_mutex);
            auto it = requests_status.find(request_id);
            if (it != requests_status.end()) {
                auto &client_id = it->second;
                // 发送完成响应
                co_await upstream.async_send(
                    boost::system::error_code{}, 
                    std::make_tuple(client_id, request_id, -1, true),
                    asio::use_awaitable);
                requests_status.erase(it);
            }
        }
    }

    asio::awaitable<void> gather(client_id_t client_id, RequestMessage &req)
    {
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            requests_status[req.request_id] = client_id;
        }

        // 简化的请求分发
        int cid = std::max(0, std::min(req.cluster_id, static_cast<int>(clusters.size()) - 1));
        for (auto &shard : clusters[cid]) {
            co_spawn(co_await asio::this_coro::executor, shard->gather(req), detached);
        }
    }

private:
    std::mutex requests_mutex;
    std::unordered_map<request_id_t, client_id_t> requests_status;
    std::vector<std::vector<std::shared_ptr<TPShard>>> clusters;
    BlockTable &bt;
    GroupChannel downstream;  // 这个需要在初始化列表中初始化
    ZMQChannel &upstream;
};

// 简化的响应发送器
asio::awaitable<void> response_sender(ZmqCoroutineSocket &router_socket, ZMQChannel &response_channel)
{
    try {
        while (true) {
            auto message = co_await response_channel.async_receive(asio::use_awaitable);
            auto client_id = std::get<0>(message);
            auto request_id = std::get<1>(message);
            auto layer_id = std::get<2>(message);
            auto success = std::get<3>(message);

            ResponseMessage response = {request_id, layer_id, success};
            std::stringstream buffer;
            msgpack::pack(buffer, response);
            std::string response_data = buffer.str();

            std::vector<zmq::message_t> response_messages;
            response_messages.emplace_back(client_id.data(), client_id.size());
            response_messages.emplace_back(response_data.data(), response_data.size());

            co_await router_socket.async_send_multipart(std::move(response_messages));
        }
    } catch (const std::exception &e) {
        std::cout << "Response sender stopped: " << e.what() << std::endl;
    }
}

asio::awaitable<void> router_receiver(ZmqCoroutineSocket &router_socket, TPGroup &group)
{
    while (true) {
        try {
            auto msg = co_await router_socket.async_recv_multipart();
            if (msg && msg->size() == 2) {
                std::vector<zmq::message_t> messages = std::move(*msg);
                const auto *data0 = static_cast<const uint8_t *>(messages[0].data());
                client_id_t client_id(data0, data0 + messages[0].size());

                const char *data1 = static_cast<const char *>(messages[1].data());
                std::string request_data(data1, data1 + messages[1].size());

                msgpack::object_handle handle = msgpack::unpack(request_data.data(), request_data.size());
                RequestMessage request;
                handle.get().convert(request);

                co_spawn(co_await asio::this_coro::executor, group.gather(client_id, request), detached);
            }
        } catch (const std::exception &e) {
            std::cerr << "Receiver error: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    try {
        Config config = parse_arguments(argc, argv);
        BlockTable bt(config);
        asio::io_context &io_context = config.get_io_context();

        g_program_start_time = std::chrono::steady_clock::now();

        // 启动服务器
        std::vector<std::shared_ptr<Server>> server_list;
        for (auto &endpoint : config.server_list) {
            server_list.emplace_back(std::make_shared<Server>(io_context, endpoint, bt, config.num_layers, config.zmq_port));
            co_spawn(io_context, server_list.back()->run(), detached);
        }

        // 启动客户端处理
        if (!config.shard_list.empty() || !config.shard_clusters.empty()) {
            ZMQChannel response_channel(io_context, 128);
            ZmqCoroutineSocket zmq_router(ZMQ_ROUTER, io_context);
            
            auto tp_group = std::make_shared<TPGroup>(config, bt, response_channel);
            
            std::string address = "tcp://*:" + std::to_string(config.zmq_port);
            zmq_router.bind(address);

            co_spawn(io_context, tp_group->run(), detached);
            co_spawn(io_context, router_receiver(zmq_router, *tp_group), detached);
            co_spawn(io_context, response_sender(zmq_router, response_channel), detached);
            co_spawn(io_context, print_statistics(), detached);

            std::cout << "Omni Xfer started. ZMQ: " << address << std::endl;
        }

        // 运行IO上下文
        std::vector<std::thread> threads;
        for (size_t i = 0; i < config.num_threads; ++i) {
            threads.emplace_back([&io_context]() { io_context.run(); });
        }
        io_context.run();

        for (auto &thread : threads) thread.join();
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}