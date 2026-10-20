/**
 * @file        graphics/resolve_readback_guard.cpp
 *
 * @brief       readback_resolve = "lazy": resolve results are copied back to guest
 *              memory only when the CPU accesses them.
 *
 * @license     BSD 3-Clause License
 */

#include <rex/graphics/resolve_readback_guard.h>

#include <algorithm>
#include <chrono>

#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>
#include <rex/system/xmemory.h>
#include <rex/thread.h>

#if REX_PLATFORM_WIN32
#include <windows.h>
#endif

namespace rex::graphics {

namespace {

// Guest views of physical memory: 64 KB, 16 MB and 4 KB pages. The host address of physical
// address P in a view is virtual_membase + view + P (Memory::MapViews; the 4 KB view's 0x1000
// offset is in its guest addresses, not in the host mapping).
constexpr uint32_t kPhysicalViews[] = {0xA0000000u, 0xC0000000u, 0xE0000000u};
constexpr uint32_t kPhysicalViewSize = 0x20000000;
constexpr uint32_t kHostPageSizeLog2 = 12;
constexpr uint32_t kHostPageSize = uint32_t(1) << kHostPageSizeLog2;
constexpr uint32_t kPhysicalPageCount = kPhysicalViewSize >> kHostPageSizeLog2;

// The vectored handler is process-wide, so there can be only one active instance.
std::atomic<ResolveReadbackGuard*> g_active_guard{nullptr};

// Protection generations of the physical pages, bumped by OnProtectionChange (on any thread) after
// rex::memory has changed the protection of a page in any of the views. Static, so that a change
// reported while the guard shuts down doesn't touch freed memory.
std::atomic<uint8_t*> g_observed_membase{nullptr};
std::atomic<uint32_t> g_protection_generations[kPhysicalPageCount];
// Bumped by changes of unknown extent (a whole allocation released in the views).
std::atomic<uint32_t> g_protection_epoch{0};

// First physical page of a range and the page after its last one.
constexpr uint32_t FirstPage(uint32_t start) { return start >> kHostPageSizeLog2; }
constexpr uint32_t EndPage(uint32_t start, uint32_t length) {
  return uint32_t((uint64_t(start) + length + kHostPageSize - 1) >> kHostPageSizeLog2);
}

// How long a CPU access waits for the command processor thread before going on with whatever is in
// guest memory (only if that thread is stuck - normally it's well under a frame).
constexpr auto kWaitTimeout = std::chrono::seconds(2);

}  // namespace

ResolveReadbackGuard::~ResolveReadbackGuard() { Shutdown(); }

bool ResolveReadbackGuard::Initialize(memory::Memory& memory, std::atomic<bool>& requests_pending,
                                      rex::thread::Event& wakeup) {
#if REX_PLATFORM_WIN32
  if (active_) {
    return true;
  }
  ResolveReadbackGuard* expected = nullptr;
  if (!g_active_guard.compare_exchange_strong(expected, this)) {
    REXGPU_WARN("readback_resolve lazy: another command processor uses it already");
    return false;
  }
  virtual_membase_ = memory.virtual_membase();
  requests_pending_ = &requests_pending;
  wakeup_ = &wakeup;
  command_processor_thread_id_ = GetCurrentThreadId();
  vectored_handler_ = AddVectoredExceptionHandler(1, VectoredHandler);
  if (!vectored_handler_) {
    g_active_guard.store(nullptr);
    REXGPU_WARN("readback_resolve lazy: AddVectoredExceptionHandler failed");
    return false;
  }
  // Before anything is guarded: every later protection change must be seen.
  guarded_generation_.assign(kPhysicalPageCount, 0);
  guarded_epoch_ = g_protection_epoch.load(std::memory_order_acquire);
  g_observed_membase.store(virtual_membase_, std::memory_order_release);
  rex::memory::SetProtectionChangeObserver(OnProtectionChange);
  active_ = true;
  REXGPU_INFO("readback_resolve lazy: resolve destinations are copied back on CPU access");
  return true;
#else
  (void)memory;
  (void)requests_pending;
  (void)wakeup;
  return false;
#endif
}

void ResolveReadbackGuard::Shutdown() {
#if REX_PLATFORM_WIN32
  if (!active_) {
    return;
  }
  RemoveVectoredExceptionHandler(vectored_handler_);
  vectored_handler_ = nullptr;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const Range& range : armed_) {
    Unguard(range.start, range.length);
  }
  for (const Range& range : requested_) {
    Unguard(range.start, range.length);
  }
  armed_.clear();
  requested_.clear();
  rex::memory::SetProtectionChangeObserver(nullptr);
  g_observed_membase.store(nullptr, std::memory_order_release);
  guarded_generation_.clear();
  guarded_generation_.shrink_to_fit();
  active_ = false;
  serviced_ticket_ = request_ticket_;
  serviced_.notify_all();
  g_active_guard.store(nullptr);
#endif
}

void ResolveReadbackGuard::Arm(uint32_t start, uint32_t length) {
  if (!active_ || !length || start >= kPhysicalViewSize) {
    return;
  }
  uint32_t end = start + std::min(length, kPhysicalViewSize - start);
  std::lock_guard<std::mutex> lock(mutex_);
  // Keep the armed ranges disjoint: a CPU access gets all the data overlapping its page back.
  for (size_t i = 0; i < armed_.size();) {
    const Range& range = armed_[i];
    if (Overlaps(range, start, end - start)) {
      start = std::min(start, range.start);
      end = std::max(end, range.start + range.length);
      armed_[i] = armed_.back();
      armed_.pop_back();
    } else {
      ++i;
    }
  }
  armed_.push_back({start, end - start});
  // Pages guarded already (the destination was resolved before and the CPU didn't touch it) are
  // skipped without a system call.
  Guard(start, end - start);
}

uint64_t ResolveReadbackGuard::TakeRequests(std::vector<Range>& ranges) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (requested_.empty() && request_ticket_ == serviced_ticket_) {
    return 0;
  }
  ranges.insert(ranges.end(), requested_.begin(), requested_.end());
  in_service_.insert(in_service_.end(), requested_.begin(), requested_.end());
  requested_.clear();
  return request_ticket_;
}

void ResolveReadbackGuard::Complete(uint64_t ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const Range& range : in_service_) {
    Unguard(range.start, range.length);
  }
  // Unguarding works on whole pages: guard again the pages armed ranges share with what was copied.
  for (const Range& armed : armed_) {
    uint32_t armed_first = armed.start & ~(kHostPageSize - 1);
    uint32_t armed_end = (armed.start + armed.length + kHostPageSize - 1) & ~(kHostPageSize - 1);
    for (const Range& range : in_service_) {
      if (Overlaps(range, armed_first, armed_end - armed_first)) {
        Guard(armed.start, armed.length);
        break;
      }
    }
  }
  in_service_.clear();
  serviced_ticket_ = std::max(serviced_ticket_, ticket);
  serviced_.notify_all();
}

void ResolveReadbackGuard::OnGuardPageAccess(uint32_t physical_address, bool is_write) {
  (void)is_write;
  uint32_t page = physical_address & ~(kHostPageSize - 1);
  std::unique_lock<std::mutex> lock(mutex_);
  if (!active_) {
    return;
  }
  // The exception has taken PAGE_GUARD off the page in the view accessed.
  guarded_generation_[FirstPage(page)] = 0;
  bool wait = false;
  for (size_t i = 0; i < armed_.size();) {
    if (Overlaps(armed_[i], page, kHostPageSize)) {
      requested_.push_back(armed_[i]);
      armed_[i] = armed_.back();
      armed_.pop_back();
      wait = true;
    } else {
      ++i;
    }
  }
  if (!wait) {
    // Another access may be waiting for the same data already.
    for (const std::vector<Range>* ranges : {&requested_, &in_service_}) {
      for (const Range& range : *ranges) {
        if (Overlaps(range, page, kHostPageSize)) {
          wait = true;
        }
      }
    }
  }
  if (!wait) {
    // A guard left from a range copied back meanwhile.
    return;
  }
#if REX_PLATFORM_WIN32
  if (GetCurrentThreadId() == command_processor_thread_id_) {
    // It can't wait for itself (and doesn't read guest memory through the virtual views normally).
    // The request stays and is serviced at the next packet.
    if (!stat_own_thread_++) {
      REXGPU_WARN("readback_resolve lazy: guarded physical page {:08X} accessed by the command "
                  "processor thread",
                  page);
    }
    return;
  }
#endif
  uint64_t ticket = ++request_ticket_;
  requests_pending_->store(true, std::memory_order_release);
  wakeup_->Set();
  auto wait_start = std::chrono::steady_clock::now();
  bool serviced = serviced_.wait_for(lock, kWaitTimeout, [this, ticket]() {
    return serviced_ticket_ >= ticket || !active_;
  });
  uint64_t wait_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - wait_start)
                                  .count());
  ++stat_waits_;
  stat_wait_us_max_ = std::max(stat_wait_us_max_, wait_us);
  if (!serviced) {
    ++stat_timeouts_;
    REXGPU_WARN("readback_resolve lazy: physical page {:08X} not copied back in {} ms, going on "
                "with the data in guest memory ({} timeouts)",
                page, wait_us / 1000, stat_timeouts_);
  }
}

void ResolveReadbackGuard::Guard(uint32_t start, uint32_t length) {
  uint32_t epoch = g_protection_epoch.load(std::memory_order_acquire);
  if (epoch != guarded_epoch_) {
    std::fill(guarded_generation_.begin(), guarded_generation_.end(), 0);
    guarded_epoch_ = epoch;
  }
  uint32_t page_end = std::min(EndPage(start, length), kPhysicalPageCount);
  uint32_t run_first = page_end;
  for (uint32_t page = FirstPage(start); page <= page_end; ++page) {
    if (page < page_end) {
      // The generation is taken before the protection is checked and changed: if it changes
      // meanwhile, the page is checked again next time.
      uint32_t generation = g_protection_generations[page].load(std::memory_order_acquire) + 1;
      if (guarded_generation_[page] != generation) {
        guarded_generation_[page] = generation;
        if (run_first == page_end) {
          run_first = page;
        }
        continue;
      }
    }
    if (run_first != page_end) {
      if (!SetGuard(run_first << kHostPageSizeLog2, (page - run_first) << kHostPageSizeLog2,
                    true)) {
        std::fill(guarded_generation_.begin() + run_first, guarded_generation_.begin() + page, 0);
      }
      run_first = page_end;
    }
  }
}

void ResolveReadbackGuard::Unguard(uint32_t start, uint32_t length) {
  uint32_t page_end = std::min(EndPage(start, length), kPhysicalPageCount);
  uint32_t page_first = std::min(FirstPage(start), page_end);
  std::fill(guarded_generation_.begin() + page_first, guarded_generation_.begin() + page_end, 0);
  SetGuard(start, length, false);
}

bool ResolveReadbackGuard::SetGuard(uint32_t start, uint32_t length, bool guard) {
  bool succeeded = true;
#if REX_PLATFORM_WIN32
  uint32_t first = start & ~(kHostPageSize - 1);
  uint64_t end = (uint64_t(start) + length + kHostPageSize - 1) & ~uint64_t(kHostPageSize - 1);
  for (uint32_t view : kPhysicalViews) {
    uint8_t* address = virtual_membase_ + view + first;
    uint8_t* view_end = virtual_membase_ + view + end;
    while (address < view_end) {
      MEMORY_BASIC_INFORMATION info;
      if (!VirtualQuery(address, &info, sizeof(info))) {
        succeeded = false;
        break;
      }
      uint8_t* region_end =
          std::min(view_end, static_cast<uint8_t*>(info.BaseAddress) + info.RegionSize);
      DWORD protect = info.Protect;
      // Pages the guest can't access in this view are left alone: real access violations.
      if (info.State == MEM_COMMIT && (protect & 0xFF) && !(protect & PAGE_NOACCESS)) {
        DWORD new_protect = guard ? (protect | PAGE_GUARD) : (protect & ~DWORD(PAGE_GUARD));
        if (new_protect != protect) {
          DWORD old_protect;
          if (!VirtualProtect(address, size_t(region_end - address), new_protect, &old_protect)) {
            succeeded = false;
          }
        }
      }
      address = region_end;
    }
  }
#else
  (void)start;
  (void)length;
  (void)guard;
#endif
  return succeeded;
}

void ResolveReadbackGuard::OnProtectionChange(void* base_address, size_t length) {
  uint8_t* membase = g_observed_membase.load(std::memory_order_acquire);
  if (!membase) {
    return;
  }
  uintptr_t begin = reinterpret_cast<uintptr_t>(base_address);
  uintptr_t views_begin = reinterpret_cast<uintptr_t>(membase) + kPhysicalViews[0];
  uintptr_t views_end = reinterpret_cast<uintptr_t>(membase) + 0x100000000ull;
  if (!length) {
    if (begin >= views_begin && begin < views_end) {
      g_protection_epoch.fetch_add(1, std::memory_order_release);
    }
    return;
  }
  uintptr_t end = begin + length;
  if (end <= views_begin || begin >= views_end) {
    return;
  }
  for (uint32_t view : kPhysicalViews) {
    uintptr_t view_begin = reinterpret_cast<uintptr_t>(membase) + view;
    uintptr_t changed_begin = std::max(begin, view_begin);
    uintptr_t changed_end = std::min(end, view_begin + kPhysicalViewSize);
    if (changed_begin >= changed_end) {
      continue;
    }
    uint32_t page_end =
        uint32_t((changed_end - view_begin + kHostPageSize - 1) >> kHostPageSizeLog2);
    for (uint32_t page = uint32_t((changed_begin - view_begin) >> kHostPageSizeLog2);
         page < page_end; ++page) {
      g_protection_generations[page].fetch_add(1, std::memory_order_release);
    }
  }
}

#if REX_PLATFORM_WIN32
long __stdcall ResolveReadbackGuard::VectoredHandler(PEXCEPTION_POINTERS exception_pointers) {
  const EXCEPTION_RECORD* record = exception_pointers->ExceptionRecord;
  if (record->ExceptionCode != STATUS_GUARD_PAGE_VIOLATION || record->NumberParameters < 2) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  ResolveReadbackGuard* guard = g_active_guard.load(std::memory_order_acquire);
  if (!guard) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  uintptr_t host_address = uintptr_t(record->ExceptionInformation[1]);
  uintptr_t membase = reinterpret_cast<uintptr_t>(guard->virtual_membase_);
  if (host_address < membase + 0xA0000000ull || host_address >= membase + 0x100000000ull) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  uint64_t offset = host_address - membase;
  uint32_t view = offset >= 0xE0000000ull   ? 0xE0000000u
                  : offset >= 0xC0000000ull ? 0xC0000000u
                                            : 0xA0000000u;
  guard->OnGuardPageAccess(uint32_t(offset - view), record->ExceptionInformation[0] == 1);
  // The page has lost PAGE_GUARD with the exception: the access is repeated with the page's own
  // protection (a write to GPU-written memory then goes to the write watch as usual).
  return EXCEPTION_CONTINUE_EXECUTION;
}
#endif

}  // namespace rex::graphics
