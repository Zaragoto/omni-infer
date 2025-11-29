// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <boost/asio.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>
#include <ox_block_table.hpp>
#include <ox_request_manager.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;
using namespace boost::asio::experimental::awaitable_operators;

struct ZmqCmd {
    table_id_t table_id = 0;
    size_t layer_id = 0;
    block_list_t block_ids;
    int rank = 0;

    MSGPACK_DEFINE_MAP(table_id, layer_id, block_ids, rank);
};

inline void optimize_tcp_socket(tcp::socket &socket)
{
    boost::system::error_code ec;

    socket.set_option(tcp::no_delay(true), ec);
    if (ec)
        std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;

#ifdef TCP_QUICKACK
    int quickack = 1;
    setsockopt(socket.native_handle(), IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
#endif
}

class Session : public std::enable_shared_from_this<Session> {
public:
    Session(tcp::socket socket, BlockTable &bt, int num_layers, std::shared_ptr<ZmqCoroutineSocket> shared_zmq)
        :  socket_(std::move(socket)), bt_(bt), num_layers_(num_layers), zmq_socket_(shared_zmq), io_ctx_(static_cast<asio::io_context&>(socket_.get_executor().context()))
    {}

    void start()
    {
        co_spawn(
            socket_.get_executor(),
            [self = shared_from_this(), this]() -> asio::awaitable<void> { co_await self->process_connection(num_layers_); },
            asio::detached);

        co_spawn(
            socket_.get_executor(),
            [self = shared_from_this(), this]() -> asio::awaitable<void> { co_await self->send_layers(); },
            asio::detached);
    }

private:
    asio::awaitable<void> process_connection(int num_layers)
    {
        try {
            while (true) {
                int64_t uid = 0;
                co_await async_read(socket_, asio::buffer(&uid, sizeof(uid)), use_awaitable);
                std::cerr << "[Session]: received uid from D = " << uid << std::endl;

                std::shared_ptr<Request> req;
                {
                    std::lock_guard<std::mutex> lock(global_requests_mutex);
                    auto it = global_requests.find(uid);
                    if (it != global_requests.end()) {
                        req = it->second;
                    }
                }

                if (!req) {
                    auto new_req = std::make_shared<Request>(uid, num_layers);
                    {
                        std::lock_guard<std::mutex> req_lock(new_req->req_mutex);
                        for (auto& [layer_id, layer_state] : new_req->layer_states) {
                            layer_state.ready = false;
                            layer_state.sent = false;
                            layer_state.need = true;
                            layer_state.sending = false;
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(global_requests_mutex);
                        global_requests[uid] = new_req;
                    }
                    req = new_req;
                } else {
                    std::lock_guard<std::mutex> req_lock(req->req_mutex);
                    for (auto &[layer_id, layer_state] : req->layer_states) {
                        if(!layer_state.sent){
                            layer_state.need = true;
                        }
                    }
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Connection closed: " << e.what() << std::endl;
        }
    }

    asio::awaitable<void> send_layers() {
        auto executor = co_await boost::asio::this_coro::executor;

        for (;;) {
            std::vector<int64_t> req_to_delete;

            {   
                std::lock_guard<std::mutex> lock(global_requests_mutex);

                for (auto& [uid, req_ptr] : global_requests) {
                    if (!req_ptr) continue;

                    bool all_sent = true;

                    std::lock_guard<std::mutex> req_lock(req_ptr->req_mutex);

                    for (auto &[layer_id, state] : req_ptr->layer_states) {
                        if (layer_id > 0) {
                            auto it_prev = req_ptr->layer_states.find(layer_id - 1);
                            if (it_prev == req_ptr->layer_states.end() || !it_prev->second.sent) {
                                continue;
                            }
                        }

                        if (!state.sent)
                            all_sent = false;

                        // 修复：确保状态正确且没有正在发送
                        if (state.ready && !state.sent && state.need && !state.sending && !state.buffers.empty()) {
                            std::cerr << "[Server] Sending layer " << layer_id << " for uid " << uid << std::endl;
                            
                            state.sending = true;
                            
                            // 使用co_spawn避免阻塞发送循环
                            co_spawn(
                                executor,
                                [self = shared_from_this(), state_ptr = &state, layer_id = layer_id, uid = uid]() -> asio::awaitable<void> {
                                    co_await state_ptr->send(self->socket_, layer_id);
                                },
                                asio::detached
                            );
                        }
                    }

                    if (all_sent) {
                        req_to_delete.push_back(uid);
                    }
                }
            }

            for (auto uid : req_to_delete) {
                remove_request(uid);
            }

            co_await boost::asio::steady_timer(executor, std::chrono::milliseconds(10))
                .async_wait(asio::use_awaitable);
        }
    }

    tcp::socket socket_;
    BlockTable &bt_;
    int num_layers_;
    std::shared_ptr<ZmqCoroutineSocket> zmq_socket_;
    asio::io_context& io_ctx_;
};

class Server {
private:
    struct ZmqTask {
        ZmqCmd cmd;
        int64_t uid;
    };
public:
    Server(asio::io_context &io_context, const tcp::endpoint &ep, BlockTable &bt, int num_layers, int zmq_port = 0)
        : acceptor_(io_context, ep), bt_(bt), num_layers_(num_layers), zmq_port_(zmq_port), shared_zmq_(nullptr),
          io_context_(io_context),
          zmq_task_channel_(io_context, 1000)
    {
        std::cout << "[Server] OX TCP Port:" << ep.port() << std::endl;

        if(zmq_port_ > 0) {
            shared_zmq_ = std::make_shared<ZmqCoroutineSocket>(ZMQ_PULL, io_context);
            std::string addr = "tcp://*:" + std::to_string(zmq_port_);
            shared_zmq_->bind(addr);
            std::cerr << "[Server ZMQ] bound at " << addr << std::endl;
        }
    }

    asio::awaitable<void> run()
    {
        if (shared_zmq_) {
            co_spawn(
                acceptor_.get_executor(),
                [this]() -> asio::awaitable<void> { co_await listen_zmq_global(); },
                asio::detached);

            co_spawn(
                acceptor_.get_executor(),
                [this]() -> asio::awaitable<void> { co_await process_zmq_tasks(); },
                asio::detached);
        }

        while (true) {
            try {
                tcp::socket socket = co_await acceptor_.async_accept(asio::use_awaitable);
                optimize_tcp_socket(socket);

                std::cout << "[Server] New TCP from: " << socket.remote_endpoint().address().to_string() << ":"
                          << socket.remote_endpoint().port() << "\n";
                std::make_shared<Session>(std::move(socket), bt_, num_layers_, shared_zmq_)->start();
            } catch (const boost::system::system_error &e) {
                std::cerr << "Accept error: " << e.what() << std::endl;
                if (e.code() == boost::asio::error::operation_aborted) {
                    break;
                }
            }
        }
        co_return;
    }

    asio::awaitable<void> listen_zmq_global()
    {
        std::cerr << "[Session ZMQ] trying ..." << std::endl;
        if (!shared_zmq_)
            co_return;

        auto& zmq_socket = *shared_zmq_;
    
        for (;;) {  
            try {
                auto msg_opt = co_await zmq_socket.async_recv_multipart();
                if (!msg_opt) {
                    continue;
                }

                std::vector<zmq::message_t> msgs = std::move(*msg_opt);
                if (msgs.empty()) {
                    continue;
                }

                std::string payload(static_cast<const char*>(msgs.front().data()),
                                    msgs.front().size());

                try {
                    msgpack::object_handle oh = msgpack::unpack(payload.data(), payload.size());
                    ZmqCmd cmd;
                    oh.get().convert(cmd);
                
                    std::cerr << "[Session ZMQ] table=" << cmd.table_id << " layer=" << cmd.layer_id
                              << " blocks=" << cmd.block_ids.size() << " rank=" << cmd.rank << std::endl;
                
                    int64_t uid = generate_uid_from_block_list(cmd.block_ids);
                    std::cerr << "[Session]: get uid from zmq: " << uid << std::endl;

                    ZmqTask task{std::move(cmd), uid};
                    co_await zmq_task_channel_.async_send(boost::system::error_code{}, task, asio::use_awaitable);
                
                } catch (const std::exception& e) {
                    std::cerr << "[Session ZMQ] parse error: " << e.what() << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Session ZMQ] listener exception: " << e.what() << std::endl;
                // 修复：避免在catch块内使用co_await
                asio::steady_timer timer(io_context_, std::chrono::milliseconds(1));
                timer.async_wait([this](boost::system::error_code) {
                    // 继续处理
                });
            }
        }
    }

private:
    asio::awaitable<void> process_zmq_tasks()
    {
        for (;;) {
            try {
                auto task = co_await zmq_task_channel_.async_receive(asio::use_awaitable);
                handle_zmq_task(task);
            } catch (const boost::system::system_error &e) {
                    std::cerr << "[Task Processor] Channel error: " << e.what() << std::endl;
                    continue;
            } catch (const std::exception& e) {
                std::cerr << "[Task Processor] Exception: " << e.what() << std::endl;
            }
        }
    }
    
    void handle_zmq_task(const ZmqTask& task)
    {
        std::shared_ptr<Request> req;
        {
            std::lock_guard<std::mutex> lock(global_requests_mutex);
            auto it = global_requests.find(task.uid);
            if (it != global_requests.end()) {
                req = it->second;
            }
        }

        if (!req) {
            auto new_req = std::make_shared<Request>(task.uid, num_layers_);
            
            {
                std::lock_guard<std::mutex> lock(new_req->req_mutex);
                auto& target_layer = new_req->layer_states[task.cmd.layer_id];
                target_layer.ready = true;
                target_layer.buffers = bt_.get_buffers_one_layer(
                    /*table_id*/ 0,
                    /*block_list*/ task.cmd.block_ids,
                    /*rank*/ 0,
                    /*layer_id*/ task.cmd.layer_id);
                    
                if (!target_layer.buffers.empty()) {
                    std::cerr << "[Server]: get buffers of layer " << task.cmd.layer_id << std::endl;
                } else {
                    std::cerr << "[Server]: empty buffers of layer " << task.cmd.layer_id << std::endl;
                }
            }
            
            add_request(new_req);
        } else {
            std::lock_guard<std::mutex> lock(req->req_mutex);
            auto& target_layer = req->layer_states[task.cmd.layer_id];
            if (!target_layer.ready) {
                target_layer.buffers = bt_.get_buffers_one_layer(
                    /*table_id*/ 0,
                    /*block_list*/ task.cmd.block_ids,
                    /*rank*/ 0,
                    /*layer_id*/ task.cmd.layer_id);
                    
                if (!target_layer.buffers.empty()) {
                    std::cerr << "[Server]: get buffers of layer " << task.cmd.layer_id << std::endl;
                } else {
                    std::cerr << "[Server]: empty buffers of layer " << task.cmd.layer_id << std::endl;
                }
                target_layer.ready = true;
            }
        }
    }

private:
    tcp::acceptor acceptor_;
    BlockTable &bt_;
    int num_layers_;
    int zmq_port_;
    std::shared_ptr<ZmqCoroutineSocket> shared_zmq_;
    asio::io_context &io_context_;
    asio::experimental::concurrent_channel<void(boost::system::error_code, ZmqTask)> zmq_task_channel_;
};