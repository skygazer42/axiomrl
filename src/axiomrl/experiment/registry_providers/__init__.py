from axiomrl.experiment.registry_providers.actor_critic import ACTOR_CRITIC_SPECS
from axiomrl.experiment.registry_providers.contrib import CONTRIB_SPECS
from axiomrl.experiment.registry_providers.goal_conditioned import GOAL_CONDITIONED_SPECS
from axiomrl.experiment.registry_providers.offline import OFFLINE_SPECS
from axiomrl.experiment.registry_providers.on_policy import ON_POLICY_SPECS
from axiomrl.experiment.registry_providers.value_based import VALUE_BASED_SPECS
from axiomrl.experiment.registry_providers.world_model import WORLD_MODEL_SPECS

ALL_SPEC_GROUPS = {
    "on_policy": ON_POLICY_SPECS,
    "value_based": VALUE_BASED_SPECS,
    "actor_critic": ACTOR_CRITIC_SPECS,
    "offline": OFFLINE_SPECS,
    "world_model": WORLD_MODEL_SPECS,
    "goal_conditioned": GOAL_CONDITIONED_SPECS,
    "contrib": CONTRIB_SPECS,
}


__all__ = [
    "ACTOR_CRITIC_SPECS",
    "ALL_SPEC_GROUPS",
    "CONTRIB_SPECS",
    "GOAL_CONDITIONED_SPECS",
    "OFFLINE_SPECS",
    "ON_POLICY_SPECS",
    "VALUE_BASED_SPECS",
    "WORLD_MODEL_SPECS",
]
