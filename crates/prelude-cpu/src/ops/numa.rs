//! Pure Rust NUMA-aware thread initialization.
//!
//! Detects physical CPU cores on NUMA node 0 and pins rayon worker threads
//! to them, achieving the same effect as SGLang's `init_cpu_threads_env`
//! without requiring libnuma or libtorch.
//!
//! Also sets NUMA memory policy to prefer the local node (via `set_mempolicy`
//! syscall) so allocations stay close to the pinned cores.

use std::collections::HashSet;

/// Detect physical CPU core IDs on NUMA node 0.
///
/// Reads `/sys/devices/system/node/node0/cpulist` and filters out hyperthreads
/// by checking `thread_siblings_list`.
/// Returns `None` if the sysfs topology is unavailable (e.g. non-Linux).
pub fn detect_numa0_physical_cores() -> Option<Vec<usize>> {
    let cpulist = std::fs::read_to_string("/sys/devices/system/node/node0/cpulist").ok()?;
    let all_cpus = parse_cpu_range(cpulist.trim());
    if all_cpus.is_empty() {
        return None;
    }

    let mut physical_cores: Vec<usize> = Vec::new();
    let mut seen_siblings: HashSet<usize> = HashSet::new();

    for &cpu in &all_cpus {
        if seen_siblings.contains(&cpu) {
            continue;
        }
        let siblings_path = format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
            cpu
        );
        if let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) {
            let siblings = parse_cpu_range(siblings_str.trim());
            for &sib in &siblings {
                seen_siblings.insert(sib);
            }
            if let Some(&first) = siblings.first() {
                if all_cpus.contains(&first) {
                    physical_cores.push(first);
                }
            }
        } else {
            physical_cores.push(cpu);
        }
    }

    if physical_cores.is_empty() {
        return None;
    }
    physical_cores.sort();
    physical_cores.dedup();
    Some(physical_cores)
}

/// Detect ALL physical CPU core IDs across all NUMA nodes.
///
/// Reads `/sys/devices/system/cpu/online` and filters out hyperthreads.
/// Returns `None` if sysfs topology is unavailable.
pub fn detect_all_physical_cores() -> Option<Vec<usize>> {
    let cpulist = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let all_cpus = parse_cpu_range(cpulist.trim());
    if all_cpus.is_empty() {
        return None;
    }

    let mut physical_cores: Vec<usize> = Vec::new();
    let mut seen_siblings: HashSet<usize> = HashSet::new();

    for &cpu in &all_cpus {
        if seen_siblings.contains(&cpu) {
            continue;
        }
        let siblings_path = format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
            cpu
        );
        if let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) {
            let siblings = parse_cpu_range(siblings_str.trim());
            for &sib in &siblings {
                seen_siblings.insert(sib);
            }
            if let Some(&first) = siblings.first() {
                physical_cores.push(first);
            }
        } else {
            physical_cores.push(cpu);
        }
    }

    if physical_cores.is_empty() {
        return None;
    }
    physical_cores.sort();
    physical_cores.dedup();
    Some(physical_cores)
}

/// Initialize rayon global thread pool with NUMA-pinned threads.
///
/// 1. Detects physical cores on NUMA node 0
/// 2. Sets NUMA memory policy to bind to node 0
/// 3. Builds rayon global pool with threads pinned to those cores
///
/// Must be called **once**, early in `main()`, before any rayon work.
/// Returns a report string, or an error message.
pub fn init_numa_rayon_pool() -> String {
    let cores = match detect_numa0_physical_cores() {
        Some(c) if !c.is_empty() => c,
        _ => {
            return "NUMA: topology not available, using default rayon pool".to_string();
        }
    };

    let n_threads = cores.len();

    // Set NUMA memory policy: MPOL_BIND to node 0
    set_mempolicy_bind_node0();

    // Build rayon global pool with pinned threads
    let cores_clone = cores.clone();
    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .spawn_handler(move |thread| {
            let cpu_id = cores_clone[thread.index() % cores_clone.len()];
            std::thread::Builder::new()
                .name(format!("rayon-numa-{}", thread.index()))
                .spawn(move || {
                    pin_current_thread_to_core(cpu_id);
                    thread.run();
                })
                .map(|_| ())
        })
        .build_global();

    match result {
        Ok(()) => {
            format!(
                "NUMA: rayon pool = {} threads pinned to node 0 cores {:?}",
                n_threads, &cores
            )
        }
        Err(e) => {
            // Global pool already initialized (e.g. by a test or prior call)
            format!("NUMA: rayon pool already initialized: {}", e)
        }
    }
}

// ── Linux syscall helpers ───────────────────────────────────────────────

/// Pin the calling thread to a specific CPU core via `sched_setaffinity`.
pub(crate) fn pin_current_thread_to_core(cpu_id: usize) {
    #[cfg(target_os = "linux")]
    {
        use nix::sched::{sched_setaffinity, CpuSet};
        use nix::unistd::Pid;
        let mut cpuset = CpuSet::new();
        if cpuset.set(cpu_id).is_ok() {
            let _ = sched_setaffinity(Pid::from_raw(0), &cpuset);
        }
    }
}

/// Set NUMA memory policy to MPOL_BIND on node 0.
/// This restricts future allocations to NUMA node 0 memory.
pub(crate) fn set_mempolicy_bind_node0() {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: SYS_set_mempolicy is a well-defined Linux syscall.
        // nodemask lives on the stack for the duration of the syscall.
        // nix does not wrap set_mempolicy, so we use the raw syscall.
        let nodemask: nix::libc::c_ulong = 1; // node 0 → bitmask = 0x1
        unsafe {
            nix::libc::syscall(
                nix::libc::SYS_set_mempolicy,
                2 as nix::libc::c_int, // MPOL_BIND
                &nodemask as *const nix::libc::c_ulong,
                std::mem::size_of::<nix::libc::c_ulong>() * 8 + 1,
            );
        }
    }
}

// ── Parsing helpers ──────────────────────────────────────────────────────

/// Parse CPU range string like "0-55" or "0,1,2,3" or "0-3,8-11".
fn parse_cpu_range(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>()) {
                result.extend(s..=e);
            }
        } else if let Ok(n) = part.parse::<usize>() {
            result.push(n);
        }
    }
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_range() {
        assert_eq!(parse_cpu_range("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_range("0,1,2,3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_range("0-3,8-11"), vec![0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(parse_cpu_range("5"), vec![5]);
        assert_eq!(parse_cpu_range(""), Vec::<usize>::new());
    }

    #[test]
    fn test_detect_numa0_physical_cores() {
        // This test just verifies the function doesn't panic; result depends on hardware
        let result = detect_numa0_physical_cores();
        if let Some(cores) = &result {
            assert!(!cores.is_empty(), "detected cores should not be empty");
            // Cores should be sorted and unique
            for w in cores.windows(2) {
                assert!(w[0] < w[1], "cores should be sorted and unique");
            }
        }
        eprintln!("Detected NUMA0 physical cores: {:?}", result);
    }
}
