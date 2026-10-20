#pragma once
/**
 * @file        graphics/resolve_readback_guard.h
 *
 * @brief       readback_resolve = "lazy": resolve results are copied back to guest
 *              memory only when the CPU accesses them.
 *
 * @license     BSD 3-Clause License
 */

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rex::memory {
class Memory;
}  // namespace rex::memory

namespace rex::thread {
class Event;
}  // namespace rex::thread

namespace rex::graphics {

// Most resolve destinations are only read by the GPU (render targets reused as textures), and
// copying every resolve back to guest memory ("fast") costs a large share of the command processor
// thread and of the bus. In the "lazy" mode a resolve leaves its data in GPU memory, and the pages
// of the destination get PAGE_GUARD in the guest views of physical memory (0xA0000000, 0xC0000000,
// 0xE0000000). The first CPU access to such a page (read or write) waits until the command
// processor thread has copied the GPU-written pages of that destination back, then continues. The
// CPU gets the current data, as with "full", without stalling on resolves nobody reads.
//
// The command processor thread arms destinations after resolves, and services requests: it takes
// them, copies the data and completes them. The data copied back is limited to the pages the GPU
// copy is still newer for (SharedMemory::ForEachGpuWrittenRange), so a range freed or written by
// the CPU meanwhile is left alone.
//
// Most destinations are resolved again every frame and stay guarded. Checking their protection
// with VirtualQuery would cost more than the readback saves (it walks the page table entries of the
// whole region: ~65 us per call for a few MB, 3 views, ~1500 resolves per second), so the guard
// remembers which pages it has guarded instead: every page protection change made through
// rex::memory bumps a generation counter of the physical page (SetProtectionChangeObserver), and
// a page counts as guarded while its generation is the one recorded when it was guarded. A guard
// page exception or unguarding clears the record.
//
// Windows only for now (PAGE_GUARD and a vectored exception handler): Initialize fails elsewhere,
// and the backend uses "fast" instead.
class ResolveReadbackGuard {
 public:
  struct Range {
    uint32_t start;
    uint32_t length;
  };

  ResolveReadbackGuard() = default;
  ResolveReadbackGuard(const ResolveReadbackGuard&) = delete;
  ResolveReadbackGuard& operator=(const ResolveReadbackGuard&) = delete;
  ~ResolveReadbackGuard();

  // On the command processor thread. One instance per process can be active. A CPU access sets
  // requests_pending and signals wakeup to get the command processor thread to service it.
  bool Initialize(memory::Memory& memory, std::atomic<bool>& requests_pending,
                  rex::thread::Event& wakeup);
  void Shutdown();
  bool is_active() const { return active_; }

  // Command processor thread: the GPU has written [start, start + length) of physical memory.
  void Arm(uint32_t start, uint32_t length);

  // Command processor thread: moves the ranges CPU accesses are waiting for to `ranges` and
  // returns a ticket for Complete, or 0 if there are none.
  uint64_t TakeRequests(std::vector<Range>& ranges);
  // Command processor thread: the data of the ranges taken with the ticket is in guest memory.
  void Complete(uint64_t ticket);

 private:
  // A CPU access to a guarded page of a guest view of physical memory: waits until the data there
  // has been copied back.
  void OnGuardPageAccess(uint32_t physical_address, bool is_write);
  // Under mutex_: guards the pages of a physical range that aren't known to be guarded already.
  void Guard(uint32_t start, uint32_t length);
  // Under mutex_: removes PAGE_GUARD from the pages of a physical range.
  void Unguard(uint32_t start, uint32_t length);
  // Adds or removes PAGE_GUARD for the accessible pages of a physical range in all views. Returns
  // false if changing the protection failed somewhere.
  bool SetGuard(uint32_t start, uint32_t length, bool guard);
  static bool Overlaps(const Range& range, uint32_t start, uint32_t length) {
    return range.start < start + length && start < range.start + range.length;
  }

  // Any thread: rex::memory has changed the protection of a host range.
  static void OnProtectionChange(void* base_address, size_t length);

#if defined(_WIN32)
  static long __stdcall VectoredHandler(struct _EXCEPTION_POINTERS* exception_pointers);
#endif

  bool active_ = false;
  uint8_t* virtual_membase_ = nullptr;
  std::atomic<bool>* requests_pending_ = nullptr;
  rex::thread::Event* wakeup_ = nullptr;
  uint32_t command_processor_thread_id_ = 0;
  void* vectored_handler_ = nullptr;

  std::mutex mutex_;
  std::condition_variable serviced_;
  // Disjoint ranges with GPU data newer than guest memory and guarded pages.
  std::vector<Range> armed_;
  // Ranges CPU accesses wait for, not taken by the command processor thread yet.
  std::vector<Range> requested_;
  // Ranges the command processor thread is copying back.
  std::vector<Range> in_service_;
  uint64_t request_ticket_ = 0;
  uint64_t serviced_ticket_ = 0;
  // Per physical page: 1 + the protection generation the page was guarded (or found not guardable)
  // at, 0 if unknown.
  std::vector<uint32_t> guarded_generation_;
  // The protection epoch guarded_generation_ is for (bumped by changes of unknown extent).
  uint32_t guarded_epoch_ = 0;

  // Statistics for the log, under mutex_.
  uint64_t stat_waits_ = 0;
  uint64_t stat_wait_us_max_ = 0;
  uint64_t stat_timeouts_ = 0;
  uint64_t stat_own_thread_ = 0;
};

}  // namespace rex::graphics
