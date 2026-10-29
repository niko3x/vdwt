//! A fixed pool of worker threads for the codec's data-parallel stages (DESIGN.md §7).
//!
//! [`Pool::run`] calls a task for every index `0..n`, spread over the workers and the calling
//! thread, and returns once all are done. The threads are created once, with the encoder or
//! decoder: a frame at 60 fps has several parallel stages, and spawning threads for each would
//! cost more than some of them take.
//!
//! Results never depend on the thread count. Tasks write disjoint outputs; reductions combine
//! fixed partitions of the work in a fixed order.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

type Task = dyn Fn(usize) + Sync;

/// A task with its lifetime erased. [`Pool::run`] waits for every worker to finish with it before
/// returning, even when unwinding, so it never outlives the borrow it came from.
#[derive(Clone, Copy)]
struct TaskRef(*const Task);

// SAFETY: the task is `Sync`, and the pointer is only used while `run` keeps the task alive.
unsafe impl Send for TaskRef {}

struct State {
    /// Bumped for every batch, so workers see each one exactly once.
    batch: u64,
    task: Option<TaskRef>,
    count: usize,
    /// Workers still working on the current batch.
    busy: usize,
    panicked: bool,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
    /// The next index to hand out.
    next: AtomicUsize,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        // Tasks run outside the lock, so a panicking task cannot poison it.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Claims and runs indices until none are left.
    fn work(&self, task: TaskRef, count: usize) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= count {
                break;
            }
            // SAFETY: see `TaskRef`.
            unsafe { (*task.0)(i) };
        }
    }
}

/// Worker threads plus the calling thread.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pool running tasks on `threads` threads in all: the caller and `threads − 1` workers.
    /// 0 counts as 1.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                batch: 0,
                task: None,
                count: 0,
                busy: 0,
                panicked: false,
                shutdown: false,
            }),
            start: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
        });
        let workers = (1..threads.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("vdwt-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("cannot spawn a codec thread")
            })
            .collect();
        Self { shared, workers }
    }

    /// Threads tasks run on, the caller's included.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `task(i)` for every `i` in `0..n`, in no particular order and possibly
    /// concurrently, and returns when all calls have. A panicking task panics here once every
    /// call has finished. Not reentrant: tasks must not call `run` on the same pool.
    pub fn run(&self, n: usize, task: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() || n <= 1 {
            (0..n).for_each(task);
            return;
        }
        // SAFETY: only the lifetime changes, and `Finish` below waits for every worker to stop
        // using the task before this function returns or unwinds.
        let task: *const (dyn Fn(usize) + Sync + '_) = task;
        let task = TaskRef(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const Task>(task)
        });
        {
            let mut state = self.shared.lock();
            assert!(state.task.is_none(), "Pool::run is not reentrant");
            self.shared.next.store(0, Ordering::Relaxed);
            state.task = Some(task);
            state.count = n;
            state.busy = self.workers.len();
            state.panicked = false;
            state.batch += 1;
        }
        self.shared.start.notify_all();

        struct Finish<'a>(&'a Shared);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                let mut state = self.0.lock();
                while state.busy > 0 {
                    state = self.0.done.wait(state).unwrap_or_else(|e| e.into_inner());
                }
                state.task = None;
            }
        }
        let finish = Finish(&self.shared);
        self.shared.work(task, n);
        drop(finish);
        if self.shared.lock().panicked {
            panic!("a codec worker thread panicked");
        }
    }
}

impl Pool {
    /// `f(i, &mut items[i])` for every item, as [`Self::run`] does.
    pub fn for_each_mut<T: Send>(&self, items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        // Each item sits behind its own lock, which only its task ever takes.
        let cells: Vec<Mutex<&mut T>> = items.iter_mut().map(Mutex::new).collect();
        self.run(cells.len(), &|i| {
            let mut item = cells[i].lock().unwrap_or_else(|e| e.into_inner());
            f(i, &mut item);
        });
    }

    /// `[f(0), f(1), …, f(n − 1)]`, computed as [`Self::run`] does.
    pub fn map<R: Send>(&self, n: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
        let out: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
        self.run(n, &|i| {
            *out[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(f(i));
        });
        out.into_iter()
            .map(|m| {
                m.into_inner()
                    .unwrap_or_else(|e| e.into_inner())
                    .expect("every index ran")
            })
            .collect()
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        let (task, count) = {
            let mut state = shared.lock();
            while state.batch == seen && !state.shutdown {
                state = shared.start.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.shutdown {
                return;
            }
            seen = state.batch;
            (state.task.expect("a new batch has a task"), state.count)
        };
        let ok = catch_unwind(AssertUnwindSafe(|| shared.work(task, count))).is_ok();
        let mut state = shared.lock();
        state.panicked |= !ok;
        state.busy -= 1;
        if state.busy == 0 {
            shared.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.start.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("threads", &self.threads())
            .finish()
    }
}

/// Per-row progress of a wavefront: tasks publish how far each row is done and wait for rows
/// they depend on. A waiter spins, then yields a bounded number of times, then sleeps until a
/// publish wakes it (a publish only takes the lock when someone sleeps).
pub(crate) struct Progress {
    done: Vec<AtomicUsize>,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
}

impl Progress {
    pub fn new(rows: usize) -> Self {
        Self {
            done: (0..rows).map(|_| AtomicUsize::new(0)).collect(),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        }
    }

    /// Publishes that row `row` is done up to `value`.
    pub fn set(&self, row: usize, value: usize) {
        self.done[row].store(value, Ordering::SeqCst);
        if self.sleepers.load(Ordering::SeqCst) > 0 {
            // Taking the lock orders this with a sleeper between its check and its wait.
            drop(self.lock.lock().unwrap_or_else(|e| e.into_inner()));
            self.wake.notify_all();
        }
    }

    /// Waits until row `row` is done up to `value` at least.
    pub fn wait(&self, row: usize, value: usize) {
        let ready = || self.done[row].load(Ordering::SeqCst) >= value;
        for k in 0..SPINS + YIELDS {
            if ready() {
                return;
            }
            if k < SPINS {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        // Counted as a sleeper before the last check, so a publish after it wakes this thread.
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        while !ready() {
            guard = self.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        drop(guard);
        self.sleepers.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Checks a waiter makes spinning, then yielding, before it sleeps: most waits are short, and a
/// sleeper wakes tens of microseconds late (8–10% of a heavy encode on an x86 laptop), while
/// unbounded yielding cost the A55 a core's worth of system calls.
const SPINS: usize = 256;
const YIELDS: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn runs_every_index_once() {
        for threads in [1, 2, 4, 7] {
            let pool = Pool::new(threads);
            for n in [0, 1, 2, 5, 100, 1000] {
                let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
                pool.run(n, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
            }
        }
    }

    #[test]
    fn writes_are_visible_after_run() {
        let pool = Pool::new(4);
        let cells: Vec<Mutex<u64>> = (0..64).map(|_| Mutex::new(0)).collect();
        for round in 1..50u64 {
            pool.run(cells.len(), &|i| {
                *cells[i].lock().unwrap() = round * i as u64
            });
            for (i, c) in cells.iter().enumerate() {
                assert_eq!(*c.lock().unwrap(), round * i as u64);
            }
        }
    }

    #[test]
    fn map_and_for_each_mut() {
        let pool = Pool::new(4);
        assert_eq!(
            pool.map(100, |i| i * i),
            (0..100).map(|i| i * i).collect::<Vec<_>>()
        );
        let mut v: Vec<usize> = (0..100).collect();
        pool.for_each_mut(&mut v, |i, x| *x += i);
        assert_eq!(v, (0..100).map(|i| 2 * i).collect::<Vec<_>>());
    }

    #[test]
    fn a_panicking_task_panics_the_caller_and_the_pool_survives() {
        let pool = Pool::new(3);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(16, &|i| assert!(i != 9, "task 9 fails"));
        }));
        assert!(result.is_err());
        let sum = AtomicU64::new(0);
        pool.run(10, &|i| {
            sum.fetch_add(i as u64, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 45);
    }

    #[test]
    fn progress_wakes_waiters() {
        let progress = Progress::new(2);
        let pool = Pool::new(4);
        // Row 1 waits on row 0 at every step; row 0 publishes slowly enough to make it sleep.
        pool.run(2, &|row| {
            for k in 1..=50 {
                if row == 0 {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                    progress.set(0, k);
                } else {
                    progress.wait(0, k);
                    progress.set(1, k);
                }
            }
        });
        progress.wait(1, 50);
    }
}
