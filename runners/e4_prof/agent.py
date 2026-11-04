"""e4 profiling agent, started by wrap.sh next to the engine (same container, same CPU set).

Every second it records per-thread CPU of the engine's process tree and of aiperf (both run in
this container), per-core CPU of the allowed cores, and every 10 s RSS/thread counts, into
$RESULT_DIR/e4prof/*.csv. Rows are appended and flushed as they are taken: the job ends with a
SIGKILL, so nothing is written at exit.

The profile window opens E4_WINDOW_DELAY_S after aiperf logs the start of its profiling phase in
$RESULT_DIR/benchmark.log (not at boot: warm-up must stay out of the window) and lasts E4_WINDOW_S.
What opens it depends on the mode: torch touches the trigger the scheduler hook polls, pyspy and
gdbpmp attach to the engine processes, nsys starts and stops the session wrap.sh launched under.
"""
import argparse
import os
import re
import subprocess
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")
PHASE_RE = re.compile(r"Phase profiling \(profiling\) started")
TREE_REFRESH_S = 5
MEM_EVERY_S = 10


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _stat_fields(text):
    # comm may contain spaces and parens; fields after the last ')' are fixed (proc(5)).
    rest = text[text.rindex(")") + 2:].split()
    return rest  # rest[0] = state (field 3)


def _proc_title(pid):
    raw = _read(f"/proc/{pid}/cmdline")
    return raw.split("\0")[0] if raw else ""


def _cmdline(pid):
    raw = _read(f"/proc/{pid}/cmdline")
    return raw.replace("\0", " ").strip() if raw else ""


def _tree(root):
    """Engine tree under root (by ppid) plus aiperf processes, as {pid: (ppid, role)}."""
    ppid_of = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            text = _read(f"/proc/{name}/stat")
            if text:
                ppid_of[int(name)] = int(_stat_fields(text)[1])
    found = {root: (ppid_of.get(root, 0), "engine")}
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_of.items():
            if pid not in found and ppid in found:
                found[pid] = (ppid, "engine")
                changed = True
    # No PID namespace on these nodes: only this job's cgroup is ours (other jobs run aiperf too).
    own_cgroup = _read("/proc/self/cgroup")
    for pid, ppid in ppid_of.items():
        if pid not in found and "aiperf" in _cmdline(pid) and _read(f"/proc/{pid}/cgroup") == own_cgroup:
            found[pid] = (ppid, "aiperf")
    return found


def _allowed_cores():
    for line in (_read("/proc/self/status") or "").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            cores = set()
            for part in line.split(":", 1)[1].strip().split(","):
                lo, _, hi = part.partition("-")
                cores.update(range(int(lo), int(hi or lo) + 1))
            return cores
    return set()


def _core_times(cores):
    out = {}
    for line in (_read("/proc/stat") or "").splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            parts = line.split()
            core = int(parts[0][3:])
            if core in cores:
                out[core] = [int(x) for x in parts[1:9]]  # user nice system idle iowait irq softirq steal
    return out


class Sampler:
    def __init__(self, root, out):
        self.root = root
        self.cores = _allowed_cores()
        self.tree = {}
        self.tree_at = 0.0
        self.mem_at = 0.0
        self.last_task = {}
        self.last_core = _core_times(self.cores)
        self.seen = set()
        self.f_thr = self._open(out, "threads.csv", "t,pid,tid,dticks,core,dvcs,dnvcs")
        self.f_names = self._open(out, "names.csv", "t,pid,ppid,role,tid,comm,proc_title")
        self.f_core = self._open(out, "cores.csv", "t,core,user,nice,system,idle,iowait,irq,softirq,steal")
        self.f_mem = self._open(out, "mem.csv", "t,pid,rss_kb,threads")
        print(f"[e4prof] sampler root={root} cores={len(self.cores)} clk_tck={CLK_TCK}", flush=True)

    @staticmethod
    def _open(out, name, header):
        f = open(os.path.join(out, name), "a", buffering=1)
        if f.tell() == 0:
            f.write(header + "\n")
        return f

    def tick(self, now):
        if now - self.tree_at >= TREE_REFRESH_S:
            self.tree, self.tree_at = _tree(self.root), now
        rows = []
        for pid, (ppid, role) in self.tree.items():
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for tid in tids:
                stat = _read(f"/proc/{pid}/task/{tid}/stat")
                status = _read(f"/proc/{pid}/task/{tid}/status")
                if not stat or not status:
                    continue
                f = _stat_fields(stat)
                ticks = int(f[11]) + int(f[12])  # utime + stime (fields 14, 15)
                core = f[36]  # field 39: CPU last run on
                vcs = nvcs = 0
                for line in status.splitlines():
                    if line.startswith("voluntary_ctxt_switches:"):
                        vcs = int(line.split()[1])
                    elif line.startswith("nonvoluntary_ctxt_switches:"):
                        nvcs = int(line.split()[1])
                key = (pid, tid)
                if key not in self.seen:
                    self.seen.add(key)
                    comm = stat[stat.index("(") + 1:stat.rindex(")")].replace(",", ";")
                    title = _proc_title(pid).replace(",", ";")
                    self.f_names.write(f"{now:.3f},{pid},{ppid},{role},{tid},{comm},{title}\n")
                prev = self.last_task.get(key)
                self.last_task[key] = (ticks, vcs, nvcs)
                if prev and (ticks > prev[0] or nvcs > prev[2]):
                    rows.append(f"{now:.3f},{pid},{tid},{ticks - prev[0]},{core},{vcs - prev[1]},{nvcs - prev[2]}\n")
        self.f_thr.write("".join(rows))
        cur = _core_times(self.cores)
        self.f_core.write("".join(
            f"{now:.3f},{c}," + ",".join(str(a - b) for a, b in zip(v, self.last_core[c])) + "\n"
            for c, v in sorted(cur.items()) if c in self.last_core))
        self.last_core = cur
        if now - self.mem_at >= MEM_EVERY_S:
            self.mem_at = now
            for pid in self.tree:
                st = _read(f"/proc/{pid}/status") or ""
                vals = dict(line.split(":", 1) for line in st.splitlines() if ":" in line)
                if "VmRSS" in vals:
                    self.f_mem.write(f"{now:.3f},{pid},{vals['VmRSS'].split()[0]},{vals['Threads'].strip()}\n")


def _engine_pids(sampler, title_prefix):
    return [pid for pid, (_, role) in sampler.tree.items()
            if role == "engine" and _proc_title(pid).startswith(title_prefix)]


def _open_window(mode, args, sampler, events):
    window = int(os.environ.get("E4_WINDOW_S", "60"))
    procs = []
    if mode == "torch":
        open(os.environ["E4_TORCH_TRIGGER"], "w").close()
    elif mode == "pyspy":
        # The whole TP4 tree at 100 Hz fell 13 s behind (S2 run 41891713093); TP1-3 repeat TP0, so
        # record only the frontend side: main process (TokenizerManager in npy), detokenizer, TP0.
        targets = [args.server_pid] + _engine_pids(sampler, "sglang::detokenizer") \
            + _engine_pids(sampler, "sglang::scheduler_TP0")
        for pid in targets:
            # nonblocking keeps the engine running between samples.
            procs.append(subprocess.Popen(
                ["py-spy", "record", "--pid", str(pid), "--idle", "--nonblocking",
                 "--rate", os.environ.get("E4_PYSPY_RATE", "50"), "--duration", str(window),
                 "--format", "chrometrace", "-o", os.path.join(args.out, f"pyspy.{pid}.chrometrace.json")],
                stdout=events, stderr=events))
    elif mode == "nsys":
        # perf_event_paranoid=4 on these nodes: no CPU sampling or context-switch tracing.
        subprocess.run(["nsys", "start", "--session=e4prof", "-o", os.path.join(args.out, "nsys"),
                        "--force-overwrite=true", "--sample=none", "--cpuctxsw=none"],
                       stdout=events, stderr=events)
    elif mode == "gdbpmp":
        # Scheduler TP0 hosts the Rust listener threads in the nrust arm; the same process in npy.
        targets = _engine_pids(sampler, "sglang::scheduler_TP0") or [args.server_pid]
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "agent.py"), "--gdbpmp",
             ",".join(map(str, targets)), "--out", args.out], stdout=events, stderr=events))
    return window, procs


def _close_window(mode, events):
    if mode == "nsys":
        subprocess.run(["nsys", "stop", "--session=e4prof"], stdout=events, stderr=events)


def _gdbpmp(pids, out, samples=20, gap_s=2.0):
    for i in range(samples):
        for pid in pids:
            t = time.time()
            res = subprocess.run(["gdb", "-p", pid, "-batch", "-nx", "-ex", "set pagination off",
                                  "-ex", "thread apply all bt"], capture_output=True, text=True)
            with open(os.path.join(out, f"gdbpmp.{pid}.txt"), "a") as f:
                f.write(f"=== sample {i} wall={t:.3f} took={time.time() - t:.2f}s\n{res.stdout}\n")
        time.sleep(gap_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode")
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark-log")
    ap.add_argument("--gdbpmp")
    args = ap.parse_args()
    if args.gdbpmp:
        return _gdbpmp(args.gdbpmp.split(","), args.out)

    sampler = Sampler(args.server_pid, args.out)
    events = open(os.path.join(args.out, "events.log"), "a", buffering=1)
    delay = int(os.environ.get("E4_WINDOW_DELAY_S", "30"))
    # The results dir is reused across jobs: a log older than this agent is an earlier job's.
    started = time.time()
    log_pos, phase_at, window_end, procs = 0, None, None, []
    next_tick = time.time()
    while os.path.exists(f"/proc/{args.server_pid}"):
        now = time.time()
        sampler.tick(now)
        if phase_at is None and os.path.exists(args.benchmark_log) \
                and os.path.getmtime(args.benchmark_log) >= started:
            if os.path.getsize(args.benchmark_log) < log_pos:
                log_pos = 0
            with open(args.benchmark_log, errors="replace") as f:
                f.seek(log_pos)
                chunk = f.read()
                log_pos = f.tell()
            if PHASE_RE.search(chunk):
                phase_at = now
                events.write(f"{now:.3f} aiperf profiling phase seen\n")
        if phase_at is not None and window_end is None and mode_needs_window(args.mode) \
                and now >= phase_at + delay:
            window, procs = _open_window(args.mode, args, sampler, events)
            window_end = now + window
            events.write(f"{now:.3f} window open mode={args.mode} for {window}s\n")
        if window_end is not None and now >= window_end:
            _close_window(args.mode, events)
            events.write(f"{now:.3f} window close mode={args.mode}\n")
            window_end = float("inf")
        next_tick += 1.0
        time.sleep(max(0.0, next_tick - time.time()))
    for p in procs:
        p.poll()


def mode_needs_window(mode):
    return mode in ("torch", "pyspy", "nsys", "gdbpmp")


if __name__ == "__main__":
    main()
