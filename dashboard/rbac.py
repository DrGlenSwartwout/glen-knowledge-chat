"""Business-OS RBAC: roles, the autonomy policy matrix, actor resolution."""
import os
from dataclasses import dataclass

from dashboard.actions import READ, LOW_WRITE, MONEY_SEND, IRREVERSIBLE

# Roles
OWNER = "owner"     # Glen
OPS = "ops"         # Rae
VA = "va"           # Shaira (scoped)
AGENT = "agent"     # Justus unattended
SYSTEM = "system"   # crons / webhooks
ROLES = (OWNER, OPS, VA, AGENT, SYSTEM)

# Policy modes
AUTO = "auto"
CONFIRM = "confirm"
QUEUE = "queue"
DENY = "deny"

# Static (actor x risk tier) policy. Owner money_send is special-cased below.
POLICY = {
    OWNER:  {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: CONFIRM, IRREVERSIBLE: CONFIRM},
    OPS:    {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: CONFIRM, IRREVERSIBLE: CONFIRM},
    VA:     {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: QUEUE,   IRREVERSIBLE: DENY},
    AGENT:  {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: QUEUE,   IRREVERSIBLE: DENY},
    SYSTEM: {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: QUEUE,   IRREVERSIBLE: DENY},
}


def owner_money_threshold():
    """0 (default) = confirm every owner money action. Set to e.g. 50 after the
    manual break-in period to auto-approve owner money actions under $50."""
    try:
        return float(os.environ.get("OWNER_MONEY_AUTO_THRESHOLD", "0"))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Actor:
    role: str
    name: str = ""


def policy_for(role, risk_tier, *, amount=None, threshold=None):
    """Return the policy mode (AUTO/CONFIRM/QUEUE/DENY) for this actor and tier."""
    if role == OWNER and risk_tier == MONEY_SEND:
        thr = owner_money_threshold() if threshold is None else threshold
        if thr > 0 and amount is not None and amount < thr:
            return AUTO
        return CONFIRM
    return POLICY.get(role, {}).get(risk_tier, DENY)


def resolve_actor(console_key, *, console_secret, token=None, role_for_token=None):
    """Owner master key first (backward compatible), then optional token->role."""
    if console_secret and console_key and console_key == console_secret:
        return Actor(role=OWNER, name="owner")
    if token and role_for_token:
        role = role_for_token(token)
        if role in ROLES:
            return Actor(role=role, name=str(token)[:8])
    return None


# Justus / console actor mapping. Rae is owner; Shaira is the scoped VA.
SCOPE_ROLES = {"rae": OWNER, "shaira": VA}


def role_for_owner(owner):
    """Map a console owner string (glen/rae/shaira/...) to a role. Glen and Rae
    are owners; Shaira is the VA; any unknown scoped owner defaults to VA."""
    o = (owner or "").lower()
    if o in ("glen", "owner"):
        return OWNER
    return SCOPE_ROLES.get(o, VA)


def actor_for_scope(scope, owner_hint=""):
    """Build an Actor from an auth scope. 'admin' (or empty) -> owner. A scoped
    token 'workspace:<owner>' -> the owner's role (rae owner, shaira va)."""
    if not scope or scope == "admin":
        return Actor(role=OWNER, name="owner")
    if scope.startswith("workspace:"):
        o = scope.split(":", 1)[1]
    else:
        o = owner_hint or ""
    return Actor(role=role_for_owner(o), name=o or "scoped")
