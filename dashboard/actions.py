"""Business-OS Action Registry: every operator/agent action declared once."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Risk tiers (drive the autonomy policy in dashboard.rbac).
READ = "read"
LOW_WRITE = "low_write"
MONEY_SEND = "money_send"
IRREVERSIBLE = "irreversible"
RISK_TIERS = (READ, LOW_WRITE, MONEY_SEND, IRREVERSIBLE)


@dataclass
class Action:
    key: str                      # e.g. "finance.refund_order"
    module: str                   # e.g. "finance"
    title: str                    # human label (panel button + Justus tool)
    description: str              # one line; also the Justus tool description
    risk_tier: str               # one of RISK_TIERS
    permission: Tuple[str, ...]   # roles allowed (see dashboard.rbac.ROLES)
    executor: Callable            # (params: dict, ctx: dict) -> dict
    confirm_summary: Optional[Callable] = None  # (params) -> str
    reversible: bool = False


ACTION_REGISTRY = {}


def register_action(a: Action) -> Action:
    if a.risk_tier not in RISK_TIERS:
        raise ValueError(f"unknown risk tier: {a.risk_tier}")
    if a.key in ACTION_REGISTRY:
        raise ValueError(f"duplicate action key: {a.key}")
    ACTION_REGISTRY[a.key] = a
    return a


def action(*, key, module, title, description, risk_tier, permission,
           confirm_summary=None, reversible=False):
    """Decorator: register the decorated function as an Action's executor."""
    def deco(fn):
        register_action(Action(
            key=key, module=module, title=title, description=description,
            risk_tier=risk_tier, permission=tuple(permission), executor=fn,
            confirm_summary=confirm_summary, reversible=reversible))
        return fn
    return deco


def get_action(key):
    return ACTION_REGISTRY.get(key)


def list_actions(module=None):
    return [a for a in ACTION_REGISTRY.values()
            if module is None or a.module == module]
