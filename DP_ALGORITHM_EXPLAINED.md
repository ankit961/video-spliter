# Dynamic Programming Algorithm Explained

## The Problem We're Solving

Given a video and hundreds of possible cut points (boundaries), select a sequence of non-overlapping clips that:
1. **Cover the video** (minimize gaps)
2. **Maximize quality** (select clips with good content)
3. **Respect constraints** (duration ranges, forbidden zones)

## Why DP Instead of Greedy?

**Greedy approach** (naive):
- Pick the highest-scoring clip
- Then pick the highest-scoring non-overlapping clip
- Problem: A slightly lower-scoring clip might open up access to much higher-scoring clips later
- Result: Suboptimal solutions, gaps in coverage

**DP approach** (optimal):
- Considers all possible sequences
- Finds the globally optimal solution
- Trades off between local quality and global coverage

---

## How Our DP Works

### 1. **State Definition**

```
dp[i] = Maximum score achievable using only boundaries 0..i
last_end[i] = The end time of the last clip in the optimal solution ending at boundary i
parent[i] = Backpointer: (prev_boundary_idx, clip_that_ends_here)
```

### 2. **Transition (The Core Logic)**

For each boundary `i`, we have two choices:

**Option A: Skip this boundary**
```
dp[i] = dp[i-1]  (carry forward, no clip ends here)
```

**Option B: End a clip at boundary i**
```
For each valid clip (j → i) ending at boundary i:
    gap_penalty = distance_since_last_clip * gap_penalty_per_second
    dp[i] = max(dp[i], dp[j] + score(clip) - gap_penalty)
```

### 3. **Visualization with Example**

```
Boundaries:  0  1  2  3  4  5  6  7  (time 0s, 10s, 20s, 30s, etc.)
Candidates:
  Clip A: [0→2] (duration 20s) score=5.0
  Clip B: [1→4] (duration 30s) score=4.0
  Clip C: [3→5] (duration 20s) score=6.0
  Clip D: [4→7] (duration 30s) score=4.5

DP Computation:
─────────────────────────────────────────

i=0: dp[0] = 0 (base case: no clips yet)

i=1: dp[1] = 0 (no clips ending at 1)

i=2: 
  - Clip A (0→2): gap=0, score = dp[0] + 5.0 - 0 = 5.0
  - dp[2] = 5.0
  - parent[2] = (0, Clip A)

i=3:
  - Skip: dp[3] = dp[2] = 5.0
  - (no clips end exactly at 3)

i=4:
  - Clip B (1→4): gap from end of best-at-1 = 10s
    score = dp[1] + 4.0 - 10*0.5 = 0 + 4.0 - 5.0 = -1.0 (worse than skip)
  - Skip better: dp[4] = dp[3] = 5.0

i=5:
  - Clip C (3→5): starts after last clip (which ends at 20s)
    Previous best at boundary 3: Clip A from [0→2] ends at 20s
    gap = (30s - 20s) = 10s
    score = dp[3] + 6.0 - 10*0.5 = 5.0 + 6.0 - 5.0 = 6.0
  - dp[5] = 6.0
  - parent[5] = (3, Clip C)

i=7:
  - Clip D (4→7): follows best-at-4 (which is Clip A ending at 20s)
    gap = 10s (from 20s to 30s = wait, wrong indices)
    Actually: Clip A is [0→2] = [0s→20s]
             Clip D is [4→7] = [40s→70s]
             gap = 40s - 20s = 20s
    score = dp[4] + 4.5 - 20*0.5 = 5.0 + 4.5 - 10.0 = -0.5 (worse)
  - Skip better: dp[7] = 6.0

FINAL: Sequence is [Clip A, Clip C] with score=6.0
```

### 4. **Tail Penalty** (Why your second video ended abruptly)

After picking all clips, we penalize remaining uncovered video:

```python
tail_gap = video_end - clips[-1].end
adjusted_score = best_score - tail_gap * tail_penalty_per_second
```

**Problem**: If we choose `tail_penalty_per_second = 1.0`, then 10 seconds of tail gap costs 10.0 points.
- A clip with score 8.0 that leaves 10s gap = net 8.0 - 10.0 = -2.0 (negative!)
- DP might reject it and pick a worse clip that covers more

**Solution**: We added `enforce_tail_coverage: true` and stricter penalties, but the scoring wasn't balanced well.

---

## Why Your Second Video Ended Abruptly

### Root Causes:

1. **Insufficient tail coverage enforcement** — The DP correctly optimized locally but didn't have strong enough incentive to cover the end

2. **Boundary quality at the tail** — The last valid clip candidates might be weaker than earlier clips:
   - Fewer multi-signal boundaries near the end
   - Less transcript data (transcription is complete but VAD might miss tail sections)
   - Shot boundaries might be sparse

3. **Clip score trade-off** — Lower-scoring clips that reach further got penalized too harshly

### Why Iterations Help:
When the LLM/heuristic reviewer says "SHIFT to adjacent boundary", it:
- Picks a slightly different clip boundary
- This forces the DP to reconsider the tail
- Eventually finds a solution that covers better

---

## Improvements to Reduce Iterations

### 1. **Dynamic Tail Penalty** (Smart scaling)

```python
# Instead of fixed penalty, scale it based on how close we are to the end
def compute_tail_penalty(tail_gap, video_duration):
    if tail_gap < 5.0:  # Close to end
        return tail_gap * 0.3  # Lighter penalty
    elif tail_gap < 15.0:
        return 5.0 + (tail_gap - 5.0) * 0.8  # Medium penalty
    else:
        return 100.0  # Hard reject (too much uncovered)
```

### 2. **Enforce Minimum Final Clip Quality**

```python
# Terminal state selection
# Don't just pick best score—require final clip to have:
# - Either high score, OR
# - Must cover at least to 95% of video

best_score = -inf
for i in range(n):
    tail_gap = video_end - last_end_time[i]
    coverage = 1.0 - (tail_gap / video_end)
    
    # Either excellent score OR excellent coverage
    if best_score[i] > 5.0 or coverage > 0.95:
        # This is a valid terminal state
        adjusted = best_score[i] - tail_penalty(tail_gap)
        if adjusted > best_terminal_score:
            best_terminal_score = adjusted
            best_terminal_idx = i
```

### 3. **Bias Toward Tail-Covering Clips**

```python
# During edge generation, boost scores for clips that reach far
def score_with_reach_bonus(clip, scorer):
    base_score = scorer.score(clip)
    reach_ratio = clip.end / video_duration
    
    if reach_ratio > 0.8:  # Reaches >80% through video
        return base_score + 2.0  # Significant bonus
    elif reach_ratio > 0.6:
        return base_score + 1.0
    return base_score
```

### 4. **Multi-Pass DP with Coverage Guarantee**

```python
# Pass 1: Normal DP
solution1 = optimize_coverage(graph, scorer, config)

# Pass 2: If coverage < 95%, rerun with stricter tail penalties
if solution1.coverage_ratio < 0.95:
    strict_config = config.copy()
    strict_config.tail_penalty_per_second = 2.0
    strict_config.enforce_tail_coverage = True
    strict_config.max_tail_gap = 5.0  # Stricter
    solution2 = optimize_coverage(graph, scorer, strict_config)
    return solution2
return solution1
```

### 5. **Backward Pass DP** (Novel approach)

Instead of just forward DP, also do backward:
```
backward_dp[i] = Best score achievable starting from boundary i to end

This finds: "What's the best way to cover from NOW to video end?"

Final solution = best_forward[j] + best_backward[j+1]
This guarantees tail coverage is considered.
```

---

## Summary Table: Complexity & Guarantees

| Approach | Complexity | Optimal | Tail Coverage | Notes |
|----------|-----------|---------|----------------|-------|
| **Greedy** | O(n log n) | ❌ | ❌ | Fast, often gaps |
| **DP (current)** | O(n·k) | ✅ | ⚠️ | k = edges per boundary |
| **DP + dynamic tail** | O(n·k) | ✅ | ✅ | Recommended fix |
| **DP + backward pass** | O(n·k·2) | ✅✅ | ✅✅ | Guaranteed coverage |

Where:
- n = number of boundaries
- k = average candidates ending at each boundary (usually 3-10)

---

## Quick Wins (Do These First)

1. **Increase base_clip_score** from 8.0 to 10.0 or 12.0
   - Makes clips inherently more valuable than gaps

2. **Reduce tail_penalty_per_second** for the final clip
   - Use 0.3 instead of 1.0 for the last 10 seconds

3. **Add coverage_ratio check** after DP
   - If < 95%, automatically shift the last clip to reach further

4. **Boost scores for sentences at clip end**
   - Sentence-end boundaries that reach far get +2.0 bonus

