//! Persistent spinning thread pool for GEMM dispatch.
//!
//! Solves the #1 performance problem on many-core Xeon: rayon threads park via
//! futex between GEMM calls, causing CPU prefetch pipelines and TLB entries to
//! go cold. When woken for the next GEMM, DRAM streaming bandwidth drops from
//! ~97 GB/s (warm) to ~15 GB/s (cold), making per-layer GEMM 5-7x slower than
//! the micro-benchmark.
//!
//! This pool emulates OpenMP's persistent thread team behavior (the same
//! mechanism SGLang uses): worker threads spin-wait between dispatches, keeping
//! their microarchitectural state warm. After a configurable timeout, threads
//! park to save power during idle periods and are woken on next dispatch.
//!
//! Config:
//!   CPU_GEMM_THREADS=N   — number of pool threads (default 8)
//!   CPU_GEMM_SPIN_MS=N   — spin duration in ms before parking (default 200)

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Function signature for dispatch work.
/// Called on each active worker thread with (thread_id, num_threads, context_ptr).
///
/// # Safety
/// - Must not panic.
/// - `ctx` must remain valid for the duration of the call.
pub type DispatchFn = unsafe fn(tid: usize, n_threads: usize, ctx: *const u8);

/// Shared state between dispatcher and worker threads.
/// Hot fields are separated to different cache lines to avoid false sharing.
#[repr(C)]
struct PoolShared {
    // ── Cache line 0: written by dispatcher, read by workers ──
    /// Generation counter. Bumped by dispatcher to signal new work.
    generation: CachePadded<AtomicU64>,

    // ── Cache line 1: written by workers, read by dispatcher ──
    /// Number of workers that have completed the current generation's work.
    done_count: CachePadded<AtomicU32>,

    // ── Cache line 2: work description (written by dispatcher before gen bump) ──
    dispatch_fn: AtomicUsize,
    dispatch_ctx: AtomicUsize,
    dispatch_n: AtomicU32,

    // ── Control ──
    shutdown: AtomicBool,
    spin_iters: u64,
}

/// Cache-line padded wrapper to prevent false sharing.
#[repr(C, align(64))]
struct CachePadded<T>(T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// A fixed-size thread pool where workers spin-wait between dispatches.
///
/// Mimics OpenMP's persistent thread team with `GOMP_SPINCOUNT` behavior:
/// threads stay spinning for `spin_ms` milliseconds after completing work,
/// keeping prefetch pipelines, TLB entries, and L1/L2 caches warm.
pub struct GemmPool {
    shared: Arc<PoolShared>,
    n_threads: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl GemmPool {
    /// Create a new pool with `n_threads` worker threads.
    /// Threads are pinned to NUMA node 0 physical cores if available.
    pub fn new(n_threads: usize, spin_ms: u64) -> Self {
        // PAUSE instruction takes ~20-100 cycles on modern x86.
        // At 2 GHz: ~10-50ns per iteration. Use conservative 50ns estimate.
        // 200ms / 50ns = 4_000_000 iterations.
        let spin_iters = spin_ms.saturating_mul(20_000);

        let shared = Arc::new(PoolShared {
            generation: CachePadded(AtomicU64::new(0)),
            done_count: CachePadded(AtomicU32::new(0)),
            dispatch_fn: AtomicUsize::new(0),
            dispatch_ctx: AtomicUsize::new(0),
            dispatch_n: AtomicU32::new(0),
            shutdown: AtomicBool::new(false),
            spin_iters,
        });

        // Detect all physical cores for thread pinning (across all NUMA nodes)
        let cores = super::numa::detect_all_physical_cores().unwrap_or_default();

        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let s = Arc::clone(&shared);
            let pin_cpu = if !cores.is_empty() {
                Some(cores[tid % cores.len()])
            } else {
                None
            };
            let h = std::thread::Builder::new()
                .name(format!("gemm-{tid}"))
                .spawn(move || {
                    // Pin to physical core (same as SGLang's init_cpu_threads_env)
                    if let Some(cpu) = pin_cpu {
                        super::numa::pin_current_thread_to_core(cpu);
                    }
                    worker_loop(tid, s);
                })
                .expect("failed to spawn GEMM pool thread");
            handles.push(h);
        }

        Self {
            shared,
            n_threads,
            handles,
        }
    }

    /// Dispatch work to the pool. Blocks until all active threads complete.
    ///
    /// `f` is called as `f(thread_id, num_active_threads, ctx)` on each active
    /// worker thread. `n_threads` controls how many workers to activate
    /// (clamped to pool size).
    ///
    /// # Safety
    /// - `ctx` must remain valid until this function returns.
    /// - `f` must be safe to call concurrently from `n_threads` threads.
    #[inline]
    pub unsafe fn dispatch(&self, f: DispatchFn, ctx: *const u8, n_threads: usize) {
        let n = n_threads.min(self.n_threads);
        if n == 0 {
            return;
        }

        // Write work description (Relaxed — will be made visible by Release on gen bump)
        self.shared
            .dispatch_fn
            .store(f as usize, Ordering::Relaxed);
        self.shared
            .dispatch_ctx
            .store(ctx as usize, Ordering::Relaxed);
        self.shared
            .dispatch_n
            .store(n as u32, Ordering::Relaxed);
        self.shared.done_count.store(0, Ordering::Relaxed);

        // Release fence + bump generation → makes all writes above visible to workers
        self.shared.generation.fetch_add(1, Ordering::Release);

        // Unpark any threads that may have slept past spin timeout
        for h in &self.handles[..n] {
            h.thread().unpark();
        }

        // Spin-wait for all active workers to complete.
        // Acquire ordering ensures we see all worker writes (output buffer etc.)
        while self.shared.done_count.load(Ordering::Acquire) < n as u32 {
            std::hint::spin_loop();
        }
    }

    pub fn num_threads(&self) -> usize {
        self.n_threads
    }
}

impl Drop for GemmPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        for h in &self.handles {
            h.thread().unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Worker thread main loop: spin → execute → signal done → spin again.
fn worker_loop(tid: usize, shared: Arc<PoolShared>) {
    let mut last_gen = 0u64;
    let max_spins = shared.spin_iters;
    // yield iterations: 25% of spin time spent yielding before park.
    // Matches OpenMP's 3-phase: spin → yield → futex_wait.
    let yield_iters = (max_spins / 4).max(1);

    loop {
        // ── Wait for new generation (3-phase: spin → yield → park) ──
        let mut spins = 0u64;
        'wait: loop {
            let g = shared.generation.load(Ordering::Acquire);
            if g != last_gen {
                last_gen = g;
                break 'wait;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }

            spins += 1;
            if spins >= max_spins {
                // Phase 2: yield spin (give CPU to others, wake fast)
                for _ in 0..yield_iters {
                    std::thread::yield_now();
                    let g = shared.generation.load(Ordering::Acquire);
                    if g != last_gen {
                        last_gen = g;
                        break 'wait;
                    }
                    if shared.shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                }
                // Phase 3: park (deep sleep)
                std::thread::park();
                spins = 0;
                if shared.shutdown.load(Ordering::Relaxed) {
                    return;
                }
            } else {
                std::hint::spin_loop();
            }
        }

        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        // ── Phase 2: execute work if this thread is active ──
        let n = shared.dispatch_n.load(Ordering::Relaxed) as usize;
        if tid < n {
            let f: DispatchFn =
                unsafe { std::mem::transmute(shared.dispatch_fn.load(Ordering::Relaxed)) };
            let ctx = shared.dispatch_ctx.load(Ordering::Relaxed) as *const u8;
            unsafe { f(tid, n, ctx) };

            // Signal completion (Release makes output writes visible to dispatcher)
            shared.done_count.fetch_add(1, Ordering::Release);
        }
    }
}

/// Get the global GEMM spinning thread pool (created on first use).
pub fn gemm_pool() -> &'static GemmPool {
    static POOL: OnceLock<GemmPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = super::gemm::gemm_thread_count_pub();
        let spin_ms: u64 = std::env::var("CPU_GEMM_SPIN_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        tracing::info!(
            "GEMM pool: {} spinning threads, spin_timeout={}ms",
            n,
            spin_ms
        );
        GemmPool::new(n, spin_ms)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_pool_basic_dispatch() {
        let pool = GemmPool::new(4, 10);
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.store(0, Ordering::SeqCst);

        unsafe fn work(_tid: usize, _n: usize, _ctx: *const u8) {
            COUNTER.fetch_add(1, Ordering::SeqCst);
        }

        unsafe {
            pool.dispatch(work, std::ptr::null(), 4);
        }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 4);

        // Dispatch again — threads should still be alive
        COUNTER.store(0, Ordering::SeqCst);
        unsafe {
            pool.dispatch(work, std::ptr::null(), 2);
        }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_pool_with_context() {
        let pool = GemmPool::new(4, 10);

        #[repr(C)]
        struct Ctx {
            data: [AtomicU64; 4],
        }
        let ctx = Ctx {
            data: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        };

        unsafe fn work(tid: usize, _n: usize, ctx: *const u8) {
            let ctx = &*(ctx as *const Ctx);
            ctx.data[tid].store((tid + 1) as u64, Ordering::SeqCst);
        }

        unsafe {
            pool.dispatch(work, &ctx as *const Ctx as *const u8, 4);
        }
        for i in 0..4 {
            assert_eq!(ctx.data[i].load(Ordering::SeqCst), (i + 1) as u64);
        }
    }

    #[test]
    fn test_pool_partial_threads() {
        let pool = GemmPool::new(8, 10);
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.store(0, Ordering::SeqCst);

        unsafe fn work(_tid: usize, _n: usize, _ctx: *const u8) {
            COUNTER.fetch_add(1, Ordering::SeqCst);
        }

        // Only activate 3 of 8 threads
        unsafe {
            pool.dispatch(work, std::ptr::null(), 3);
        }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 3);
    }

    /// Test that threads wake correctly after parking (spin timeout expired).
    /// This catches bugs in the spin→yield→park→unpark cycle.
    #[test]
    fn test_pool_wake_after_park() {
        // Very short spin (1ms) so threads park quickly
        let pool = GemmPool::new(4, 1);
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        unsafe fn work(_tid: usize, _n: usize, _ctx: *const u8) {
            COUNTER.fetch_add(1, Ordering::SeqCst);
        }

        // First dispatch — threads are fresh
        COUNTER.store(0, Ordering::SeqCst);
        unsafe { pool.dispatch(work, std::ptr::null(), 4); }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 4);

        // Wait for threads to spin out and park (1ms spin + yield + park)
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Second dispatch — threads must wake from park
        COUNTER.store(0, Ordering::SeqCst);
        unsafe { pool.dispatch(work, std::ptr::null(), 4); }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 4);

        // Third dispatch immediately after — threads should still be spinning
        COUNTER.store(0, Ordering::SeqCst);
        unsafe { pool.dispatch(work, std::ptr::null(), 4); }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 4);

        // Wait again, dispatch with partial threads
        std::thread::sleep(std::time::Duration::from_millis(50));
        COUNTER.store(0, Ordering::SeqCst);
        unsafe { pool.dispatch(work, std::ptr::null(), 2); }
        assert_eq!(COUNTER.load(Ordering::SeqCst), 2);
    }

    /// Stress test: many rapid dispatches interleaved with sleeps.
    #[test]
    fn test_pool_rapid_dispatch_with_gaps() {
        let pool = GemmPool::new(8, 2); // 2ms spin
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        unsafe fn work(_tid: usize, _n: usize, _ctx: *const u8) {
            COUNTER.fetch_add(1, Ordering::SeqCst);
        }

        for round in 0..10 {
            COUNTER.store(0, Ordering::SeqCst);
            let n = (round % 8) + 1;
            unsafe { pool.dispatch(work, std::ptr::null(), n); }
            assert_eq!(COUNTER.load(Ordering::SeqCst), n as u64, "round {round} n={n}");

            // Every 3rd round, sleep to trigger park
            if round % 3 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}
