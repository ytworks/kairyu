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
