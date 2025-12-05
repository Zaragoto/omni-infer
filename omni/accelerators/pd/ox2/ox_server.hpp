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

struct ZmqCmd {
    table_id_t table_id = 0;
    size_t layer_id = 0;
    block_list_t block_ids;
    int rank = 0;

    MSGPACK_DEFINE_MAP(table_id, layer_id, block_ids, rank);
};

// Enable TCP to turn on some options that reduce latency, such as disabling Nagle.
inline void optimize_tcp_socket(tcp::socket &socket)
{
    boost::system::error_code ec;

    socket.set_option(tcp::no_delay(true), ec);
    if (ec)
        std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;

#ifdef TCP_QUICKACK
    //Quick ACK for faster TCP acknowledgment, not available on all platforms.
    int quickack = 1;
    setsockopt(socket.native_handle(), IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
#endif
}

class Session : public std::enable_shared_from_this<Session> {
private:
    tcp::socket socket_;
    BlockTable &bt_;
    int num_layers_;
    std::shared_ptr<ZmqCoroutineSocket> zmq_socket_;
    asio::io_context& io_ctx_;
    concurrent_channel<void(boost::system::error_code, std::pair<std::shared_ptr<Request>, int>)> send_queue_;
public:
    Session(tcp::socket socket, BlockTable &bt, int num_layers, std::shared_ptr<ZmqCoroutineSocket> shared_zmq)
        :  socket_(std::move(socket)), bt_(bt), num_layers_(num_layers),
           zmq_socket_(shared_zmq), io_ctx_(static_cast<asio::io_context&>(socket_.get_executor().context())),
           send_queue_(io_ctx_, 1000)
    {}

    void start()
    {
        co_spawn(
            socket_.get_executor(),
            [self = shared_from_this(), this]() -> asio::awaitable<void> {
                co_await self->process_connection(num_layers_);
            },
            detached);
        co_spawn(
            socket_.get_executor(),
            [self = shared_from_this(), this]() -> asio::awaitable<void> {
                co_await self->process_send_queue();
            },
            detached);
    }

    void try_trigger_send(const std::shared_ptr<Request>& req, int layer_id, const char* trigger_source) {
        std::lock_guard<std::mutex> lock(req->req_mutex);
        auto it = req->layer_states.find(layer_id);
        if(it == req->layer_states.end()) {
            std::cerr << "[TTS][TRY] uid = " << req->uid
                      << " layer = " << layer_id
                      << " from = " << trigger_source
                      << " but no such layer in layer_states"
                      << std::endl;
            return;
        }

        auto &state = it->second;

        std::cerr << "[TTS][TRY] uid = " << req->uid
                  << " layer = " << layer_id
                  << " from = " << trigger_source
                  << " phase = " << state.phase_str()
                  << " buffers size = " << state.buffers.size()
                  << std::endl;

        // Only those with a status of Pending can be considered for joining the queue.
        if(!state.can_enqueue()) {
            std::cerr << "[TTS][SKIP] uid = " << req->uid
                      << " layer = " << layer_id
                      << " from = " << trigger_source
                      << " not in Pending phase, skip enqueue."
                      << std::endl;
            return;
        }

        if(state.buffers.empty()) {
            std::cerr << "[TTS][SKIP] uid = " << req->uid
                      << " layer = " << layer_id
                      << " from = " << trigger_source
                      << " Pending but buffers empty, skip."
                      << std::endl;
            return;
        }

        // Transmission is allowed only after the previous layer has been sent.
        if(layer_id > 0) {
            auto it_prev = req->layer_states.find(layer_id - 1);
            if(it_prev == req->layer_states.end() || !it_prev->second.is_done()) {
                std::cerr << "[TTS][WAIT] uid = " << req->uid
                          << " layer = " << layer_id
                          << " from = " << trigger_source
                          << " prev layer not Done, wait."
                          << std::endl;
            return;
            }
        }

        // Enqueue
        state.mark_enqueued(layer_id, std::string("try_trigger_send from ") + trigger_source);

        std::cerr << "[TTS][ENQUEUE] uid = " << req->uid
                  << " layer = " << layer_id
                  << " from = " << trigger_source
                  << " push into send_queue_"
                  << std::endl;

        send_queue_.try_send(
            boost::system::error_code{},
            std::make_pair(req, layer_id)
        );
    }

private:
    asio::awaitable<void> process_connection(int num_layers)
    {
        try {
            while (true) {
                int64_t uid = 0;
                co_await async_read(socket_, asio::buffer(&uid, sizeof(uid)), use_awaitable);
                std::cerr << "[Session]: received uid from D = " << uid << std::endl;

                //-----------------------------------------------------
                // 查找 Request（不创建）
                //-----------------------------------------------------
                std::shared_ptr<Request> req;
                {
                    std::lock_guard<std::mutex> lock(global_requests_mutex);
                    auto it = global_requests.find(uid);
                    if (it != global_requests.end()) {
                        req = it->second;
                    }
                }
                //-----------------------------------------------------
                // 情况 A：未找到 => 创建新的 Request
                //-----------------------------------------------------
                if (!req) {
                    auto new_req = std::make_shared<Request>(uid, num_layers);

                    {
                        std::lock_guard<std::mutex> req_lock(new_req->req_mutex);

                        for (auto& [layer_id, layer_state] : new_req->layer_states) {
                            // ready/sent 全部设为 false
                            layer_state.phase = LayerPhase::Init;
                            layer_state.last_phase_modifier = "new request, Init";
                            layer_state.log_phase("NEW REQUEST", layer_id);

                            layer_state.after_receiving_tcp(layer_id,
                                "process_connection TCP uid = " + std::to_string(uid) + " create a new request");
                        }
                    }

                    // Binding a Session
                    new_req->owner = shared_from_this();

                    // 加入全局字典
                    {
                        std::lock_guard<std::mutex> lock(global_requests_mutex);
                        global_requests[uid] = new_req;
                    }

                    req = new_req;

                    // Try to send immediately
                    for(auto &[layer_id, state] : req->layer_states) {
                        try_trigger_send(req, layer_id, "tcp_new_req");
                    }
                }
                //-----------------------------------------------------
                // 情况 B：已存在 => 更新 need
                //-----------------------------------------------------
                else {
                    {
                        std::lock_guard<std::mutex> req_lock(req->req_mutex);

                        // Binding a Session
                        if(req->owner.expired()) {
                            req->owner = shared_from_this();
                        }

                        std::cerr << "[TCP][EXIST] uid = " << uid
                                  << " set Need on all non-Done layers."
                                  << std::endl;

                        for (auto &[layer_id, layer_state] : req->layer_states) {
                            if(!layer_state.is_done()) {
                                layer_state.after_receiving_tcp(layer_id,
                                "process_connection TCP uid = " + std::to_string(uid) + " exist request");
                            }
                        }
                    }

                    // Try to send immediately
                    for(auto &[layer_id, _] : req->layer_states) {
                        try_trigger_send(req, layer_id, "tcp_exist_req");
                    }
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Connection closed: " << e.what() << std::endl;
        }
    }

    /*
    asio::awaitable<void> send_layers() {
        auto executor = co_await boost::asio::this_coro::executor;

        for (;;) {
            std::vector<int64_t> req_to_delete;

            {   // === 锁 global_requests ===
                std::lock_guard<std::mutex> lock(global_requests_mutex);

                for (auto& [uid, req_ptr] : global_requests) {
                    if (!req_ptr) continue;

                    bool all_sent = true;

                    // 锁该 request
                    std::lock_guard<std::mutex> req_lock(req_ptr->req_mutex);

                    for (auto &[layer_id, state] : req_ptr->layer_states) {
                        // 如果上一层没传，先不传该层
                        
                        if (layer_id > 0) {
                            auto it_prev = req_ptr->layer_states.find(layer_id - 1);
                            if (it_prev == req_ptr->layer_states.end() || !it_prev->second.sent) {
                                continue;
                            }
                        }
                        

                        if (!state.sent)
                            all_sent = false;

                        if (state.ready && !state.sent && state.need && !state.buffers.empty() && !state.sending) {
                            state.need = false;
                            state.last_need_modifier = "第" + std::to_string(layer_id) + "层加入发送队列后清除need";
                            // ======== 把该层送进队列 ========
                            std::cerr << "\n检测到需要发送的层" << layer_id << std::endl
                                      << "目前其状态为: ready = " << state.ready
                                      << ", sent = " << state.sent
                                      << ", need = " << state.need
                                      << ", sending = " << state.sending << std::endl
                                      << "last_ready_modifier = " << state.last_ready_modifier << std::endl
                                      << "last_sent_modifier = " << state.last_sent_modifier << std::endl
                                      << "last_need_modifier = " << state.last_need_modifier << std::endl
                                      << "last_sending_modifier = " << state.last_sending_modifier << std::endl;
                            // co_spawn(
                            // executor,
                            // state.send(socket_, layer_id),     // 调用该 layer 的 send()
                            // detached
                            // );
                            // co_await state.send(socket_, layer_id);
                            co_await send_queue_.async_send(
                                boost::system::error_code{},
                                std::make_pair(req_ptr, layer_id),
                                use_awaitable
                            );
                        }
                    }

                    // 如果所有层都 sent → 删除该 request
                    if (all_sent) {
                        req_to_delete.push_back(uid);
                    }
                }
            } // === global lock 结束 ===

            // 删除 Request（不能在锁内删除）
            for (auto uid : req_to_delete) {
                remove_request(uid);
            }

            // 防止忙等
            co_await boost::asio::steady_timer(executor, std::chrono::milliseconds(10))
                .async_wait(use_awaitable);
        }
    }
    */

    asio::awaitable<void> process_send_queue() {
        for (;;) {
            auto [req, layer_id] = co_await send_queue_.async_receive(use_awaitable);

            {
                std::lock_guard<std::mutex> lock(req->req_mutex);
                auto &state = req->layer_states[layer_id];

                std::cerr << "[SEND][BEGIN] uid = " << req->uid
                          << " layer = " << layer_id
                          << " phase = " << state.phase_str()
                          << std::endl;

                state.mark_sending(layer_id, "process_send_queue begin");
            }
            
            auto &state = req->layer_states[layer_id];
            co_await state.send(socket_, layer_id);

            bool need_trigger_next = false;
            int next_layer = -1;
            bool all_done = false;
            bool need_retry = false;


            {
                std::lock_guard<std::mutex> lock(req->req_mutex);

                if(state.phase == LayerPhase::Done) {
                    std::cerr << "[SEND][OK] uid = " << req->uid
                              << " layer = " << layer_id
                              << " done, try trigger next"
                              << std::endl;

                    // if trigger next layer
                    next_layer = layer_id + 1;
                    if(next_layer < num_layers_) {
                        need_trigger_next = true;
                        // try_trigger_send(req, next_layer, "prev done");
                    }

                    // check if all layers are sent
                    all_done = true;
                    for(auto &[lid, st] : req->layer_states) {
                        if(!st.is_done()) {
                            all_done = false;
                            break;
                        }
                    }
                    if(all_done) {
                        std::cerr << "[REQ][GC] uid = " << req->uid
                                  << " all layers done, remove request"
                                  << std::endl;
                        remove_request(req->uid);
                    }
                } else if(state.phase == LayerPhase::Failed) {
                    std::cerr << "[SEND][FAILED] uid = " << req->uid
                              << " layer = " << layer_id
                              << " mark need & retry later"
                              << std::endl;

                    need_retry = true;
                    // state.after_receiving_tcp(layer_id, "send failed, retry need");
                    // try_trigger_send(req, layer_id, "retry_after_fail");
                } else {
                    std::cerr << "[SEND][UNEXPECTED] uid = " << req->uid
                              << " layer = " << layer_id
                              << " phase = " << state.phase_str()
                              << " after send. No action."
                              << std::endl;
                }
            }

            if(need_trigger_next) {
                try_trigger_send(req, next_layer, "prev done");
            }

            if(need_retry) {
                {
                    std::lock_guard<std::mutex> lock(req->req_mutex);
                    state.after_receiving_tcp(layer_id, "send failed, retry need");
                }
                try_trigger_send(req, layer_id, "retry_after_fail");
            }
        }
    }
};

class Server {
private:
    // 任务结构体
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
                detached);

            // 运行异步任务处理器
            co_spawn(
                acceptor_.get_executor(),
                [this]() -> asio::awaitable<void> { co_await process_zmq_tasks(); },
                detached);
        }

        while (true) {
            try {
                tcp::socket socket = co_await acceptor_.async_accept(use_awaitable);
                optimize_tcp_socket(socket);

                std::cout << "[Server] New TCP from: " << socket.remote_endpoint().address().to_string() << ":"
                          << socket.remote_endpoint().port() << "\n";
                std::make_shared<Session>(std::move(socket), bt_, num_layers_, shared_zmq_)->start();
                // std::make_shared<Session>(std::move(socket), bt_, num_layers_, /* no ZMQ here */ nullptr)->start();
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
                
                    std::cerr << "[Session ZMQ] table = " << cmd.table_id << " layer = " << cmd.layer_id
                              << " blocks = " << cmd.block_ids.size() << " rank = " << cmd.rank << std::endl;
                
                    int64_t uid = generate_uid_from_block_list(cmd.block_ids);
                    std::cerr << "[Session]: get uid from zmq: " << uid << std::endl;

                    // 快速创建任务并放入队列，立即返回继续接收下一条消息
                    ZmqTask task{std::move(cmd), uid};
                    co_await zmq_task_channel_.async_send(boost::system::error_code{}, task, use_awaitable);
                
                } catch (const std::exception& e) {
                    std::cerr << "[Session ZMQ] parse error: " << e.what() << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Session ZMQ] listener exception: " << e.what() << std::endl;
                asio::steady_timer timer(co_await asio::this_coro::executor, std::chrono::milliseconds(1));
                co_await timer.async_wait(use_awaitable);
            }
        }
    }

private:
    asio::awaitable<void> process_zmq_tasks()
    {
        for (;;) {
            try {
                auto task = co_await zmq_task_channel_.async_receive(use_awaitable);
                // 处理任务
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
                
                target_layer.buffers = bt_.get_buffers_one_layer(
                    /*table_id*/ 0,
                    /*block_list*/ task.cmd.block_ids,
                    /*rank*/ 0,
                    /*layer_id*/ task.cmd.layer_id);

                // ===== [P] 打印该 layer 的 buffers 大小 =====
                std::size_t p_total_bytes = 0;
                std::cerr << "[P][BUF] uid = " << task.uid
                          << " layer = " << task.cmd.layer_id
                          << " buffers.size = " << target_layer.buffers.size();

                for (std::size_t i = 0; i < target_layer.buffers.size(); ++i) {
                    auto sz = boost::asio::buffer_size(target_layer.buffers[i]);
                    p_total_bytes += sz;
                    std::cerr << " buf[" << i << "]=" << sz;
                }
                std::cerr << " total=" << p_total_bytes << std::endl;

                target_layer.after_receiving_zmq(
                    static_cast<int>(task.cmd.layer_id),
                    "ZMQ new data uid = " + std::to_string(task.uid));
            }
            
            add_request(new_req);
            // try_trigger_send(new_req, static_cast<int>(task.cmd.layer_id), "zmq_new");
        } else {
            int layer_id = static_cast<int>(task.cmd.layer_id);

            {
                std::lock_guard<std::mutex> lock(req->req_mutex);
                auto& target_layer = req->layer_states[layer_id];

                target_layer.buffers = bt_.get_buffers_one_layer(
                    /*table_id*/ 0,
                    /*block_list*/ task.cmd.block_ids,
                    /*rank*/ 0,
                    /*layer_id*/ task.cmd.layer_id);
                // ===== [P] 打印该 layer 的 buffers 大小 =====
                std::size_t p_total_bytes = 0;
                std::cerr << "[P][BUF] uid = " << task.uid
                          << " layer = " << task.cmd.layer_id
                          << " buffers.size = " << target_layer.buffers.size();

                for (std::size_t i = 0; i < target_layer.buffers.size(); ++i) {
                    auto sz = boost::asio::buffer_size(target_layer.buffers[i]);
                    p_total_bytes += sz;
                    std::cerr << " buf[" << i << "]=" << sz;
                }
                std::cerr << " total=" << p_total_bytes << std::endl;

                target_layer.after_receiving_zmq(
                    layer_id,
                    "ZMQ exist data uid = " + std::to_string(task.uid));
            }
            
            // try_trigger_send(req, layer_id, "zmq_exist");
            if(auto session = req->owner.lock()) {
                session->try_trigger_send(req, layer_id, "zmq_exist");
            } else {
                std::cerr << "[ZMQ][WARN] uid = " << task.uid
                          << " layer = " << layer_id
                          << " has no owning Session, skip triggrt."
                          << std::endl; 
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
    // 用来异步处理任务的通道
    concurrent_channel<void(boost::system::error_code, ZmqTask)> zmq_task_channel_;
};