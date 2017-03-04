/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
// Borrowed from https://github.com/progschj/ThreadPool/blob/master/ThreadPool.h

// Copyright (c) 2012 Jakob Progsch, Václav Zeman

// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.

// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:

//    1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.

//    2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
//
//    3. This notice may not be removed or altered from any source distribution.

// Changes:
//
// Imported original copyright notice (https://github.com/progschj/ThreadPool/blob/master/COPYING) inside the file (Nicolas Vasilache Dec. 13th 2016)
// Modified into a bounded SPMCLockFreeQueue with 500us sleeps (Nicolas Vasilache Jan. 25th 2017)

#pragma once

#include <chrono>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

#include <iostream>

class ThreadPool {
 public:
  ThreadPool(size_t);
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();
  size_t size() { return workers.size(); };

 private:

  static constexpr int kNumTasks = 1 << 12;

  // need to keep track of threads so we can join them
  std::vector< std::thread > workers;
  std::array< std::function<void()>, kNumTasks > tasks;

  bool stop;
  std::atomic<long> sentinel;
  std::atomic<long> nextCandidate;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) :
    stop(false), sentinel(0), nextCandidate(0)
{
  for(size_t i = 0; i < threads; ++i)
    workers.emplace_back(
      [this]
      {
        for(;;)
        {
          std::function<void()> task;

          auto candidate = nextCandidate.load();
          auto tail = sentinel.load();
          if (candidate == tail) {
            if(this->stop) {
              return;
            }
            constexpr int usleeptime = 500;
            std::this_thread::sleep_for(std::chrono::microseconds(usleeptime));
            continue;
          }
          if (candidate > tail) {
            std::cerr << "ERROR!!!" << std::endl;
          }
          if (!nextCandidate.compare_exchange_strong(candidate, candidate + 1)) {
            continue;
          }

          this->tasks[candidate % kNumTasks]();
        }
      }
    );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    // don't allow enqueueing after stopping the pool
    if (stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }

    tasks[sentinel % kNumTasks] = [task](){ (*task)(); };
    sentinel++;
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
  stop = true;
  for(std::thread &worker: workers) {
    worker.join();
  }
}
