/**
 * @file        kernel/crt/memory.cpp
 *
 * @brief       Native memory operation hooks -- replaces recompiled PPC
 *              implementations of memcpy, memmove, memset, etc.
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 * @license     BSD 3-Clause License
 */
#include <cstdint>
#include <cstring>

#include <rex/hook.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>
#include <rex/thread/mutex.h>

namespace rex::kernel::crt {

// ---------------------------------------------------------------------------
// Writes to watched physical memory
// ---------------------------------------------------------------------------

// Guest virtual addresses from 0xA0000000 up are the views of physical memory.
constexpr uint32_t kPhysicalViewsBase = 0xA0000000;

// A write into guest physical memory that the GPU caches (textures, vertex data) faults, and the
// handler takes the global critical region, runs the invalidation callbacks and unprotects the
// watched range around the page - for one range of the destination after another. While a large
// copy runs through such memory, a thread that needs the region in between (every kernel object
// lookup: KeSetEvent, KeReleaseSemaphore, waits) may not get it until the copy is done: a guest
// loader copying textures held up the audio render thread 13-22 ms. Trigger the callbacks for
// the whole destination at once instead, before writing, as the first fault would for its range.
static void UnwatchDestination(void* dst, size_t n) {
  if (!dst || !n || n > UINT32_MAX) {
    return;
  }
  rex::memory::Memory* memory = rex::system::kernel_memory();
  const uint32_t address = memory->HostToGuestVirtual(dst);
  if (address < kPhysicalViewsBase || !memory->IsPhysicalMemoryWatched(address, uint32_t(n))) {
    return;
  }
  // The pages at the ends of the destination keep data the copy doesn't overwrite, while the
  // invalidation drops the GPU copy of whole pages. If the GPU wrote them and the data isn't in
  // guest memory yet (readback_resolve = "lazy" guards such pages), reading them gets it back first.
  const volatile uint8_t* bytes = static_cast<const volatile uint8_t*>(dst);
  if (address & 0xFFF) {
    (void)bytes[0];
  }
  if ((address + uint32_t(n)) & 0xFFF) {
    (void)bytes[n - 1];
  }
  memory->TriggerPhysicalMemoryCallbacks(rex::thread::global_critical_region::AcquireDirect(),
                                         address, uint32_t(n), true, false);
}

// ---------------------------------------------------------------------------
// Standard memory operations
// ---------------------------------------------------------------------------

static void* native_memcpy(void* dst, const void* src, size_t n) {
  UnwatchDestination(dst, n);
  return std::memcpy(dst, src, n);
}

static void* native_memmove(void* dst, const void* src, size_t n) {
  UnwatchDestination(dst, n);
  return std::memmove(dst, src, n);
}

static void* native_memset(void* dst, int val, size_t n) {
  UnwatchDestination(dst, n);
  return std::memset(dst, val, n);
}

static void* native_memchr(const void* ptr, int val, size_t n) {
  return const_cast<void*>(std::memchr(ptr, val, n));
}

// ---------------------------------------------------------------------------
// Xbox/VMX-optimized variants (same semantics, native speed)
// ---------------------------------------------------------------------------

static void* native_XMemCpy(void* dst, const void* src, size_t n) {
  UnwatchDestination(dst, n);
  return std::memcpy(dst, src, n);
}

static void* native_XMemSet(void* dst, int val, size_t n) {
  UnwatchDestination(dst, n);
  return std::memset(dst, val, n);
}

static void* native_XMemSet128(void* dst, int val, size_t n) {
  UnwatchDestination(dst, n);
  return std::memset(dst, val, n);
}

static void* native_memset_vmx(void* dst, int val, size_t n) {
  UnwatchDestination(dst, n);
  return std::memset(dst, val, n);
}

// ---------------------------------------------------------------------------
// Secure variants (return errno_t)
// ---------------------------------------------------------------------------

static int native_memcpy_s(void* dst, size_t dstsz, const void* src, size_t count) {
  if (!dst || !src || count > dstsz)
    return 22;  // EINVAL
  UnwatchDestination(dst, count);
  std::memcpy(dst, src, count);
  return 0;
}

static int native_memmove_s(void* dst, size_t dstsz, const void* src, size_t count) {
  if (!dst || !src || count > dstsz)
    return 22;  // EINVAL
  UnwatchDestination(dst, count);
  std::memmove(dst, src, count);
  return 0;
}

}  // namespace rex::kernel::crt

REX_HOOK(rexcrt_memcpy, rex::kernel::crt::native_memcpy)
REX_HOOK(rexcrt_memmove, rex::kernel::crt::native_memmove)
REX_HOOK(rexcrt_memset, rex::kernel::crt::native_memset)
REX_HOOK(rexcrt_memchr, rex::kernel::crt::native_memchr)
REX_HOOK(rexcrt_XMemCpy, rex::kernel::crt::native_XMemCpy)
REX_HOOK(rexcrt_XMemSet, rex::kernel::crt::native_XMemSet)
REX_HOOK(rexcrt_XMemSet128, rex::kernel::crt::native_XMemSet128)
REX_HOOK(rexcrt_memset_vmx, rex::kernel::crt::native_memset_vmx)
REX_HOOK(rexcrt_memcpy_s, rex::kernel::crt::native_memcpy_s)
REX_HOOK(rexcrt_memmove_s, rex::kernel::crt::native_memmove_s)
