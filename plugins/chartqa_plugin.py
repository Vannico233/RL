"""Load with --external_plugins on both swift rollout and swift rlhf."""

from importlib.metadata import version

from swift.rewards.orm import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.callbacks import TrainerCallback, callbacks_map

from chart_rl.swift_adapter import build_scheduler_class, reward_batch
from chart_rl.training_audit import build_callback_class, record_rewards


if version("ms-swift") != "4.6.0":
    raise RuntimeError("This plugin targets ms-swift 4.6.0; install the pinned framework before training")


class ChartQAReward(ORM):
    def __call__(self, completions, solution, **kwargs):
        rewards = reward_batch(completions, solution)
        record_rewards(rewards, kwargs)
        return rewards


orms["chartqa_answer"] = ChartQAReward
multi_turns["chartqa_crop"] = build_scheduler_class(MultiTurnScheduler)
callbacks_map["chart_audit"] = build_callback_class(TrainerCallback)
