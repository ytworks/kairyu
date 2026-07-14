import pytest

from kairyu.orchestration.budget import Budget, BudgetState


def test_defaults():
    budget = Budget()
    assert budget.max_steps == 16
    assert budget.max_refine_depth == 2
    assert budget.max_cost_usd is None


def test_charge_returns_new_state_and_original_unchanged():
    state = BudgetState(budget=Budget(max_steps=2))
    charged = state.charge()
    assert charged is not state
    assert charged.steps_used == 1
    assert state.steps_used == 0


def test_exhaustion_by_steps():
    state = BudgetState(budget=Budget(max_steps=2)).charge().charge()
    assert state.is_exhausted is True
    assert state.charge().steps_used == 3  # charging never raises


def test_exhaustion_by_cost():
    state = BudgetState(budget=Budget(max_cost_usd=1.0)).charge(cost=0.6)
    assert state.is_exhausted is False
    assert state.charge(cost=0.5).is_exhausted is True


def test_can_refine_bounded_by_depth():
    state = BudgetState(budget=Budget(max_refine_depth=2))
    assert state.can_refine(depth=0) is True
    assert state.can_refine(depth=1) is True
    assert state.can_refine(depth=2) is False


def test_exhausted_state_cannot_refine():
    state = BudgetState(budget=Budget(max_steps=1)).charge()
    assert state.can_refine(depth=0) is False


def test_try_reserve_returns_new_state_when_complete_steps_fit():
    state = BudgetState(budget=Budget(max_steps=3))

    reserved = state.try_reserve(steps=2)

    assert reserved is not None
    assert reserved is not state
    assert reserved.steps_reserved == 2
    assert reserved.steps_used == 0
    assert state.steps_reserved == 0


def test_try_reserve_refuses_complete_steps_without_mutating_original():
    state = BudgetState(budget=Budget(max_steps=2)).charge()

    refused = state.try_reserve(steps=2)

    assert refused is None
    assert state.steps_used == 1
    assert state.steps_reserved == 0


def test_unknown_cost_reservation_is_exclusive_until_success_reconciliation():
    state = BudgetState(budget=Budget(max_steps=3, max_cost_usd=1.0))

    reserved = state.try_reserve(unknown_cost=True)

    assert reserved is not None
    assert reserved.unknown_cost_reserved is True
    assert reserved.try_reserve(unknown_cost=True) is None

    committed = reserved.reconcile_success(cost=0.25, unknown_cost=True)

    assert committed.unknown_cost_reserved is False
    assert committed.try_reserve(unknown_cost=True) is not None


def test_unknown_cost_reservation_refuses_exhausted_cost_cap():
    state = BudgetState(budget=Budget(max_cost_usd=1.0)).charge(cost=1.0)

    assert state.is_exhausted is True
    assert state.try_reserve(unknown_cost=True) is None


def test_release_restores_reserved_steps_and_unknown_cost_slot():
    state = BudgetState(budget=Budget(max_steps=2, max_cost_usd=1.0))
    reserved = state.try_reserve(steps=2, unknown_cost=True)
    assert reserved is not None

    released = reserved.release(steps=2, unknown_cost=True)

    assert released.steps_reserved == 0
    assert released.unknown_cost_reserved is False
    assert released.steps_used == 0
    assert released.cost_used == 0.0
    assert released.try_reserve(steps=2, unknown_cost=True) is not None


def test_success_reconciliation_commits_steps_and_actual_cost_exactly_once():
    state = BudgetState(budget=Budget(max_steps=2, max_cost_usd=1.0))
    reserved = state.try_reserve(steps=2, unknown_cost=True)
    assert reserved is not None

    committed = reserved.reconcile_success(
        steps=2, cost=0.75, unknown_cost=True
    )

    assert committed.steps_reserved == 0
    assert committed.unknown_cost_reserved is False
    assert committed.steps_used == 2
    assert committed.cost_used == 0.75
    assert reserved.steps_used == 0
    assert reserved.cost_used == 0.0
    with pytest.raises(ValueError):
        committed.reconcile_success(steps=2, cost=0.75, unknown_cost=True)


def test_reservation_transitions_reject_negative_values_and_underflow():
    state = BudgetState(budget=Budget(max_steps=2, max_cost_usd=1.0))

    with pytest.raises(ValueError):
        state.try_reserve(steps=-1)
    with pytest.raises(ValueError):
        state.release(steps=-1)
    with pytest.raises(ValueError):
        state.release()
    with pytest.raises(ValueError):
        state.reconcile_success()

    reserved = state.try_reserve(unknown_cost=True)
    assert reserved is not None
    with pytest.raises(ValueError):
        reserved.reconcile_success(cost=-0.1, unknown_cost=True)


def test_exhaustion_and_refinement_include_in_flight_reservations():
    step_state = BudgetState(budget=Budget(max_steps=1, max_refine_depth=2))
    step_reserved = step_state.try_reserve()
    assert step_reserved is not None
    assert step_reserved.steps_used == 0
    assert step_reserved.is_exhausted is True
    assert step_reserved.can_refine(depth=0) is False

    cost_state = BudgetState(
        budget=Budget(max_steps=3, max_refine_depth=2, max_cost_usd=1.0)
    )
    cost_reserved = cost_state.try_reserve(unknown_cost=True)
    assert cost_reserved is not None
    assert cost_reserved.cost_used == 0.0
    assert cost_reserved.is_exhausted is True
    assert cost_reserved.can_refine(depth=0) is False
