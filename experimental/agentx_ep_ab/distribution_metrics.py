"""Pure count metrics; input counts must already pass accounting validation."""
import math
import statistics


def metrics(counts: list[int], owners: list[int]) -> dict:
    if len(counts) != 256 or len(owners) != 256:
        raise ValueError('Expected 256 expert counts and verified expert owners')
    if any(c < 0 for c in counts) or any(r not in range(8) for r in owners):
        raise ValueError('Invalid counts or EP8 ownership')
    total = sum(counts)
    if total == 0:
        return {'assignments': 0, 'available': False}
    loads = [sum(c for c, owner in zip(counts, owners) if owner == rank) for rank in range(8)]
    probs = [c / total for c in counts if c]
    active = [c for c in counts if c]
    return {
        'assignments': total,
        'active_experts': len(active),
        'empty_expert_fraction': 1 - len(active) / 256,
        'hottest_expert_share': max(probs),
        'expert_max_over_mean': max(counts) / (total / 256),
        'effective_experts': math.exp(-sum(p * math.log(p) for p in probs)),
        'active_expert_assignments_median': statistics.median(active),
        'active_expert_assignments_max': max(active),
        'rank_assignments': loads,
        'rank_max_over_mean': max(loads) / (total / 8),
        'rank_load_cv': statistics.pstdev(loads) / (total / 8),
        'assignment_balance_efficiency': total / (8 * max(loads)),
        'note': 'Assignment balance proxy, not measured GPU utilization or latency; EP1 projected EP8 ownership is counterfactual.'
    }
