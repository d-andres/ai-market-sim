# Feature Branch Plan: Live Simulation With Non-Blocking LLM

## Objective
Keep the world running at a fixed tick rate while LLM planning is in flight, and preserve believable actor behavior via local reflex logic.

## Scope
- Convert planning from blocking-per-tick to background async job handling.
- Add stale-plan invalidation using per-actor generation/version checks.
- Add reflex behavior when no plan is ready.
- Reduce unnecessary LLM calls (strategic check-ins rather than per-step thinking).
- Keep existing trade/conversation systems compatible.

## Safety and Extensibility Constraints
- Prefer additive changes and preserve existing action contracts (`move`, `wait`, `propose_trade`, `converse`).
- Introduce new runtime state in engine-local dictionaries first (avoid risky schema migrations during early rollout).
- Keep each phase releasable and testable independently (no cross-phase hard dependency in one PR).
- Maintain deterministic tick progression regardless of model latency.
- Design action execution to be easy to extend: new action types should require minimal engine edits.

## Phase 1: Non-Blocking Planning Pipeline (Engine)

### Changes
1. Add a persistent planning executor on engine init.
2. Add per-actor planning runtime state (stored in engine dictionaries keyed by actor id):
   - pending future
   - request generation
   - current generation
   - submission tick/time (for visibility/debug)
3. Update tick loop:
   - submit planning job only if actor needs plan and has no pending job
   - do not wait for results in the same tick
   - poll completed futures each tick and apply valid plans
4. Keep execution phase running each tick using current queue/reflex fallback.
5. Add safe interrupt behavior for in-flight planning:
   - cancel and clear pending planning entries on interrupt where possible.
6. Emit lightweight planning lifecycle events (`plan_submitted`, `plan`) for observability.

### Acceptance Criteria
- Tick count advances at fixed cadence regardless of LLM latency.
- UI still updates while planning is pending.
- No blocking wait inside tick for model completion.
- Existing actions still execute correctly with no regressions.

### Rollback Strategy
- Keep legacy behavior behind small helper boundaries (submission/collection helpers).
- If instability appears, disable background submission path and fall back to synchronous planning while preserving helper methods.

## Phase 2: Plan Validity and Interrupt Safety

### Changes
1. Increment actor generation on interrupt events.
2. Capture generation at plan submission time.
3. On plan completion, discard if generation mismatches (stale plan).
4. Clear pending entries safely after apply/discard.

### Acceptance Criteria
- If an actor is interrupted while thinking, old plan is not applied.
- Event log shows stale plan discard/debug events for traceability.

## Phase 3: Reflex Layer (No LLM Required)

### Changes
1. Add engine reflex handler for actors with empty queue + pending plan:
   - wait/observe
   - adjacency handling (trade-ready/conversation-ready)
   - simple safety actions (e.g., hold position or step clear from immediate blocker)
2. Use reflex result as normal tick action summary/log entry.

### Acceptance Criteria
- Actors still act every tick even when planning is pending.
- No idle freeze due to LLM delay.

## Action Extensibility Blueprint

Target structure for maintainable growth of actions:
1. Keep action payload shape in `PlannedAction` stable (`action_type`, `params`, `reason`).
2. Move action execution dispatch toward a handler map (e.g. `{"move": _execute_move, ...}`) as a follow-up.
3. Each new action should define:
   - execution handler
   - validation expectations for `params`
   - interrupt/reflex behavior when preconditions fail
   - event log output format
4. Keep parser/prompts aligned by documenting new action schemas in one place.

## Phase 4: Replan Cadence and Load Shedding

### Changes
1. Add strategic replan cadence (periodic check-ins) instead of per-step calls.
2. Add optional cooldown before re-submit after failures.
3. Cap concurrent planning jobs if needed (queue by priority).
4. Priority order recommendation:
   - combat/survival interrupts
   - direct social interactions
   - exploration/patrol

### Acceptance Criteria
- Planning calls per minute are stable under load.
- High-priority actors/events receive planning first.

## Phase 5: Conversation Latency Optimization

### Changes
1. Keep current converse action path.
2. Add optional conversation packet mode:
   - one LLM response can include opener + short follow-up options + tone delta
3. Run short local continuation turns before requesting fresh LLM input.

### Acceptance Criteria
- Conversations progress without requiring LLM on every line.
- Relationship memory still records exchanges and impressions.

## Observability and Debugging
- Expose in world snapshot:
  - pending planner actor ids
  - pending count
  - stale plan discard count
  - plan apply count
- Add event types:
  - plan_submitted
  - plan_applied
  - plan_discarded_stale
  - reflex_action

## Data Model Approach
Use engine-local runtime state first (no schema expansion initially). If stable, migrate selected fields into actor schema later.

## Out of Scope for This Branch
- Full combat system implementation.
- Economic model redesign.
- Multi-model orchestration.

## Rollout Order (Recommended)
1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

## Definition of Done
- Simulation remains live under slow LLM response conditions.
- Actors continue to react to world changes while planning is pending.
- Stale plans never override post-interrupt reality.
- Core interactions (move/trade/converse) continue functioning.
